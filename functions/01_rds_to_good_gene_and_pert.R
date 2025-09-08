library(dplyr)
library(Seurat)
library(patchwork)
library(Matrix)
library(ggplot2)
library(reshape2)
library(pheatmap)

effect_path<-'/Users/erganshang/Desktop/diffusion_and_protein/classifier_free/PCA_emb_cleary_supergene/data/raw_related/GSE221321_RAW/'
plot_path<-'/Users/erganshang/Desktop/diffusion_and_protein/classifier_free/eff_emb_cleary_sub_cluster_saver_top_eff_gene_acc/plot/'
store_path<-'/Users/erganshang/Desktop/diffusion_and_protein/classifier_free/eff_emb_cleary_sub_cluster_saver_top_eff_gene_acc_leiden/data/embedding_related/'
data_path<-'/Users/erganshang/Desktop/diffusion_and_protein/classifier_free/PCA_emb_cleary/data/'

KO_conventional_effect_size<-read.csv(paste0(effect_path, 'GSM6858447_KO_conventional_FRPerturb_effect_sizes.csv.gz'))
KO_matrix<-acast(KO_conventional_effect_size, Downstream_gene ~ Perturbed_gene, value.var = "Log_fold_change", fill = 0)

KO_abs<-abs(KO_matrix)


pert_sum<-colSums(KO_abs)
good_pert_name<-colnames(KO_matrix)[pert_sum>=650] 
KO_matrix<-KO_matrix[, good_pert_name]
gene_sum<-rowSums(abs(KO_matrix))

effect_gene_num<-4000
effect_gene_idx<-order(gene_sum, decreasing=TRUE)[1:effect_gene_num]
effect_gene_name<-rownames(KO_matrix)[effect_gene_idx]
length(intersect(effect_gene_name, good_pert_name))
effect_gene_name<-intersect(union(effect_gene_name, good_pert_name), rownames(KO_matrix))

data<-readRDS(paste0(data_path, 'GSM6858447_KO_conventional.rds'))

count<-0
for(name in good_pert_name){
  if(name %in% rownames(data)){
    count<-count+1
  }
}


# for(name in good_gene_name){
#   vec<-data@assays$RNA@counts[name, ]
#   print(paste0('0 fraction is ', sum(vec==0)/length(vec)))
# }
# KO_submatrix = KO_matrix[good_gene_name, good_pert_name]
# p<-pheatmap(KO_submatrix,
#          color = colorRampPalette(c("blue", "white", "red"))(100),
#          cluster_rows = TRUE,
#          cluster_cols = TRUE,
#          scale = "none", 
#          show_rownames = FALSE,
#          show_colnames = FALSE,
#          fontsize = 6,
#          main = "Hierarchical Clustered KO Matrix")
# 
# ggsave(paste0(plot_path, "KO_submatrix_heatmap.png"), plot = p, width = 8, height = 6, dpi = 300)

guides_collapsed<-data@meta.data$Guides_collapsed_by_gene
cell_filter <- sapply(guides_collapsed, function(x) {
  parts <- unlist(strsplit(x, "--"))  # split "A--B" to c("A", "B")
  any(parts %in% good_pert_name) | ("non-targeting" %in% parts)  
})
sum(cell_filter)
data<-data[effect_gene_name, cell_filter] # cell in good_pert or non-targeting
guides_collapsed<-data@meta.data$Guides_collapsed_by_gene

split_cells_indices<-which(grepl("--", guides_collapsed))
data_single_pert<-data[, -split_cells_indices]

new_cell_list<-list()
new_cell_meta<-list()
indicator<-numeric(length(split_cells_indices))
count<-1
for(i in split_cells_indices){
  guide_str<-guides_collapsed[i]
  guides_split<-strsplit(guide_str, '--')[[1]]
  for(guide in guides_split){
    if(guide %in% c(good_pert_name, 'non-targeting')){
      indicator[count]<-1
      new_cell_list[[length(new_cell_list)+1]]<-data@assays$RNA@counts[, i]
      new_cell_meta[[length(new_cell_meta)+1]]<-guide
    }
  }
  count<-count+1
}
# which(indicator==0) integer(0)
new_cell_list<-do.call(rbind, new_cell_list)
new_cell_meta<-do.call(c, new_cell_meta)

new_count<-as.matrix(rbind(t(data_single_pert@assays$RNA@counts), new_cell_list))
new_pert_each_cell<-c(data_single_pert@meta.data$Guides_collapsed_by_gene, new_cell_meta)
#########################################################################################################
express_gene_num<-1500
express_gene_sum<-colSums(new_count)
express_gene_idx<-order(express_gene_sum, decreasing=TRUE)[1:express_gene_num]
express_gene_name<-colnames(new_count)[express_gene_idx]

good_gene_name<-express_gene_name
new_count<-new_count[, good_gene_name]
#########################################################################################################
frac_0<-numeric(length(good_gene_name))
names(frac_0)<-good_gene_name
for(name in good_gene_name){
  vec<-new_count[, name]
  frac_0[name]<-sum(vec==0)/length(vec)
}
sum(frac_0<=0.7) # 651 genes high expressed

library(reticulate)
np<-import("numpy")

KO_good_for_PCA<-t(KO_matrix[good_gene_name, ])
py_KO_good_for_PCA<-r_to_py(KO_good_for_PCA)
np$save(paste0(store_path, 'KO_good_for_PCA.npy'), py_KO_good_for_PCA)
save(KO_good_for_PCA, file=paste0(store_path, 'KO_good_for_PCA.Rdata'))

gene_name<-colnames(new_count)
py_gene_name<-r_to_py(gene_name)
np$save(paste0(store_path, 'good_gene_name.npy'), py_gene_name)

py_good_pert_name<-r_to_py(good_pert_name)
np$save(paste0(store_path, 'good_pert_name.npy'), py_good_pert_name)

py_new_pert_each_cell<-r_to_py(new_pert_each_cell)
np$save(paste0(store_path, 'new_pert_each_cell.npy'), py_new_pert_each_cell)

save(new_count, file=paste0(store_path, 'new_count.Rdata'))




