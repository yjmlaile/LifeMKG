import torch
from torch import nn

from . import tasks, layers
from lifemkg.base_nbfnet import BaseNBFNet
from sklearn.cluster import KMeans
from torch.nn.init import xavier_normal_
import torch.nn.functional as F
import torch.nn as nn
import einops

class LifeMKG(nn.Module):

    def __init__(self, device, dataset, entity_model_cfg):
        super(LifeMKG, self).__init__()
        self.device = device
        self.ent_nums = []
        self.rel_nums = []
        for i in range(5):
            self.ent_nums.append(dataset.snapshots[i].num_ent)
            self.rel_nums.append(dataset.snapshots[i].num_rel)

        self.RelEmbeddings = nn.Embedding(self.rel_nums[0], 64).to(self.device)
        self.GetRepresentations = GetRepresentations(entity_model_cfg.input_dim, num_mlp_layers=2)
        self.adjuster_text = self.GetRepresentations.adjuster_text
        self.adjuster_img = self.GetRepresentations.adjuster_img


        self.EntityModel = globals()[entity_model_cfg.pop('class')](**entity_model_cfg)
        self.hper=[0, 0, 0, 0.001, 0.005]
        self.current_snaps = -1
        for name, param in self.named_parameters():
            self.register_buffer(('old.' + name).replace(".", "_"), torch.tensor([[]]))
    def forward(self, entity_graph_data, batch, r_ind):

        text_feature,  img_feature = self.GetRepresentations(entity_graph_data)
        score = self.EntityModel(entity_graph_data, batch, r_ind, self.RelEmbeddings.weight.data, text_feature, img_feature)
        
        losses = 0
        if self.current_snaps > 0 and self.training:
            losses = []
            for name, param in self.named_parameters():
                if name == 'RelEmbeddings.weight':
                    old_data = getattr(self, ('old.' + name).replace(".", "_"))
                    new_data = param[:old_data.size(0)]
                    losses.append((( (new_data - old_data) ** 2).sum()))
                    
                else:
                    old_data = getattr(self, ('old.' + name).replace(".", "_"))
                    losses.append((( (param - old_data) ** 2).sum()))
            return score, self.hper[self.current_snaps]*sum(losses)
        return score, losses

    def switchsnaps(self, index):
     
        for name, param in self.named_parameters():
            value = param.data
            self.register_buffer(('old.' + name).replace(".", "_"), value.clone())
        rel_embeddings = nn.Embedding(self.rel_nums[index+1], 64).to(self.device)
        new_rel_embeddings = rel_embeddings.weight.data
        new_rel_embeddings[:self.rel_nums[index]] = torch.nn.Parameter(self.RelEmbeddings.weight.data)
        self.RelEmbeddings.weight = torch.nn.Parameter(new_rel_embeddings) 
    

class Entity(BaseNBFNet):

    def __init__(self, input_dim, hidden_dims, num_relation=1, **kwargs):
        super().__init__(input_dim, hidden_dims, num_relation, **kwargs)
        self.layers = nn.ModuleList()
        for i in range(len(self.dims) - 1):
            self.layers.append(
                layers.GeneralizedRelationalConv( 
                    self.dims[i], self.dims[i + 1], num_relation,
                    self.dims[0], self.message_func, self.aggregate_func, self.layer_norm,
                    self.activation, dependent=False, project_relations=True)
            )

        self.feature_dim = (sum(hidden_dims) if self.concat_hidden else hidden_dims[-1]) + input_dim
        self.qk = nn.Linear(64, 128)
        self.liffn = FFN(input_dim)
        self.norm = nn.LayerNorm(input_dim)
        self.mlp = nn.Sequential()
        mlp = []
        for i in range(self.num_mlp_layers - 1):
            mlp.append(nn.Linear(self.feature_dim, self.feature_dim))
            mlp.append(nn.ReLU())
        mlp.append(nn.Linear(self.feature_dim, 1))
        self.mlp = nn.Sequential(*mlp)
        
        self.mlp_320 = nn.Sequential()
        mlp_320 = []
        for i in range(self.num_mlp_layers - 1):
            mlp_320.append(nn.Linear(self.feature_dim*2 +64, self.feature_dim*2+64))
            mlp_320.append(nn.ReLU())
        mlp_320.append(nn.Linear(self.feature_dim*2+64, 64))
        self.mlp_320 = nn.Sequential(*mlp_320)


    
    def bellmanford(self, data, h_index, r_index,  rel_embeddings, text_feature, img_feature, separate_grad=False):
        batch_size = len(r_index)
        query = rel_embeddings[r_index]
        index = h_index.unsqueeze(-1).expand_as(query)
        boundary = torch.zeros(batch_size, data.num_nodes, self.dims[0], device=h_index.device)
        boundary.scatter_add_(1, index.unsqueeze(1), query.unsqueeze(1))
        
        text_feature = text_feature.unsqueeze(0).expand(batch_size, -1, -1)
        img_feature= img_feature.unsqueeze(0).expand(batch_size, -1, -1)

        size = (data.num_nodes, data.num_nodes)
        edge_weight = torch.ones(data.num_edges, device=h_index.device)
    
        hiddens = []
        edge_weights = []
        layer_input = boundary

        for i, layer in enumerate(self.layers):
            if separate_grad:
                edge_weight = edge_weight.clone().requires_grad_()
            hidden = layer(layer_input, query, boundary, data.edge_index, data.edge_type, size, edge_weight)
            if self.short_cut and hidden.shape == layer_input.shape:
                hidden = hidden + layer_input
            hiddens.append(hidden)
            edge_weights.append(edge_weight)
            layer_input = hidden
        
        m_feature = torch.cat([text_feature, img_feature, hiddens[-1]], dim=-1)
        m_query = self.mlp_320(m_feature)
        q, k = self.qk(hiddens[-1]).chunk(2, dim=-1)
        v = m_query
        all_feature = self.m_interaction(q, k, v, h_index)
        all_feature = all_feature + self.liffn(all_feature)
        all_feature = self.norm(all_feature)  

        node_query = query.unsqueeze(1).expand(-1, data.num_nodes, -1) 
        if self.concat_hidden:
            output = torch.cat(hiddens + [node_query], dim=-1)
        else:
            output_all = torch.cat([all_feature, node_query], dim=-1)
            output = torch.cat([hiddens[-1], node_query], dim=-1)
        return {
            "node_feature": output,
            "node_all_feature": output_all,
            "edge_weights": edge_weights,
        }

    def forward(self, data,  batch, r_ind,  rel_embeddings, text_feature, img_feature):
        h_index, t_index, r_index = batch.unbind(-1)
        for num_layer, layer in enumerate(self.layers):
            layer.relation = rel_embeddings
        
        if self.training:
            data = self.remove_easy_edges(data, h_index, t_index, r_index)
        shape = h_index.shape

        h_index, t_index, r_index = self.negative_sample_to_tail(h_index, t_index, r_index, r_ind,num_direct_rel=data.num_relations // 2)
        assert (h_index[:, [0]] == h_index).all()
        assert (r_index[:, [0]] == r_index).all()

        output = self.bellmanford(data, h_index[:, 0], r_index[:, 0], rel_embeddings, text_feature, img_feature)  
        feature = output["node_feature"]
        feature_all = output["node_all_feature"]
        index = t_index.unsqueeze(-1).expand(-1, -1, feature.shape[-1])
        feature = feature.gather(1, index)  
        feature_all = feature_all.gather(1, index) 
        score = self.mlp(feature).squeeze(-1)
        score.view(shape)
        score_all = self.mlp(feature_all).squeeze(-1)
        score = (score + score_all)/2
        return score



    def m_interaction(self, q, k, v, h_index):
        split = lambda t: einops.rearrange(t, 'b l (h d) -> b h l d', h=4) 
        merge = lambda t: einops.rearrange(t, 'b h l d -> b l (h d)')
        norm = lambda t: F.normalize(t, dim=-1)
        
        batch_size = q.size(0)
        num_node = q.size(1)
        
        q, k, v = map(split, [q, k, v])
        q, k = map(norm, [q, k])


        full_rank_term = torch.eye(k.size(-1)).to(h_index.device)
        full_rank_term = einops.repeat(full_rank_term, 'd D -> b h d D', b=batch_size, h=5)
        kvs = einops.einsum(k, v, 'b h v d, b h v D -> b h d D') 
        numerator =  einops.einsum(q, kvs, 'b h v d, b h d D -> b h v D') 
        numerator = numerator + einops.reduce(v, 'b h (v w) d -> b h w d', 'sum', w=1) + v*num_node
                    

        denominator = einops.einsum(q, einops.reduce(k, 'b h v d -> b h d', 'sum'), 'b h v d, b h d -> b h v')
        denominator = denominator + torch.full(denominator.shape, fill_value=num_node).to(h_index.device) + num_node
        denominator = einops.rearrange(denominator, 'b h (v w) -> b h v w', w=1)

        output = numerator / denominator
        output = merge(output)
        
        return output

class FFN(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.drop = 1
        
        self.linear1 = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.act = nn.ReLU()
        self.linear2 = nn.Linear(self.hidden_dim, self.hidden_dim)

    def forward(self, x):
        x = self.linear1(x)
        x = self.act(x)
        x = F.dropout(x, p=self.drop, training=self.training)
        x = self.linear2(x)
        return x






class GetRepresentations(torch.nn.Module):
    def __init__(self, input_dim, num_mlp_layers):
        super(GetRepresentations, self).__init__()
        self.num_mlp_layers = num_mlp_layers
        input_dim = input_dim*2  

        self.text_feature_mlp = nn.Sequential()
        text_feature_mlp = []
        for i in range(self.num_mlp_layers - 1):
            if i==0:
                text_feature_mlp.append(nn.Linear(768, input_dim))
                text_feature_mlp.append(nn.ReLU())
            else:
                text_feature_mlp.append(nn.Linear(input_dim, input_dim))
                text_feature_mlp.append(nn.ReLU())
        text_feature_mlp.append(nn.Linear(input_dim, input_dim))
        self.text_feature_mlp = nn.Sequential(*text_feature_mlp)

        self.img_feature_mlp = nn.Sequential()
        img_feature_mlp = []
        for i in range(self.num_mlp_layers - 1):
            if i==0:
                img_feature_mlp.append(nn.Linear(4096, input_dim))
                img_feature_mlp.append(nn.ReLU())
            else:
                img_feature_mlp.append(nn.Linear(input_dim, input_dim))
                img_feature_mlp.append(nn.ReLU())
        img_feature_mlp.append(nn.Linear(input_dim, input_dim))
        self.img_feature_mlp = nn.Sequential(*img_feature_mlp)
        self.adjuster_text = ClusterCentersAdjuster(input_dim, 10)
        self.adjuster_img = ClusterCentersAdjuster(input_dim, 10)

    def forward(self, data):
        text_feature = self.text_feature_mlp(data.text_features)
        text_feature, clusters_text, cluster_centers_text = self.adjuster_text(text_feature)
        img_feature = self.img_feature_mlp(data.img_features)
        img_feature, clusters_img, cluster_centers_img = self.adjuster_img(img_feature)
        return text_feature, img_feature
    
class ClusterCentersAdjuster(torch.nn.Module):
    def __init__(self, dim_ent, max_clusters=10):
        super(ClusterCentersAdjuster, self).__init__()
        self.max_clusters = max_clusters
        self.gain = torch.nn.init.calculate_gain('relu')
        self.weight = nn.Parameter(torch.rand(dim_ent, dim_ent))
        nn.init.xavier_uniform_(self.weight.data, gain=self.gain)
        self.dropout_p = 0.2

        self.vis2sem = nn.Sequential(
            nn.Linear(dim_ent, dim_ent * 2), nn.ReLU(True),
            nn.Dropout(self.dropout_p), nn.Linear(dim_ent * 2, dim_ent)
        )

        self.mlp = nn.Sequential(
            nn.Linear(dim_ent * 2, dim_ent * 2), nn.ReLU(),
            nn.Linear(dim_ent * 2, dim_ent)
        )

        self.linear = nn.Linear(dim_ent, dim_ent)
        self.dropout = nn.Dropout(self.dropout_p)
        self.norm = nn.LayerNorm(dim_ent)

        self.cached_clusters = None
        self.cached_cluster_centers = None

    def forward(self, features, force_recluster=False):
        features = self.vis2sem(features)
        device = features.device

  
        if self.cached_clusters is None or self.cached_cluster_centers is None or force_recluster:
            features_np = features.cpu().detach().numpy()
            kmeans = KMeans(n_clusters=15, random_state=0, n_init= 'auto')
            clusters = kmeans.fit_predict(features_np)
            cluster_centers = torch.Tensor(kmeans.cluster_centers_).to(device)

            self.cached_clusters = clusters
            self.cached_cluster_centers = cluster_centers
        else:
            clusters = self.cached_clusters
            cluster_centers = self.cached_cluster_centers

        cluster_centers_features = torch.matmul(cluster_centers[clusters], self.weight)

        adjusted_features = torch.cat([features, cluster_centers_features], dim=-1)
        adjusted_features = self.mlp(adjusted_features)
        adjusted_features = cluster_centers_features + torch.sigmoid(adjusted_features) * features
        adjusted_features = self.norm(self.dropout(torch.relu(self.linear(adjusted_features))) + adjusted_features)

        return adjusted_features, clusters, cluster_centers