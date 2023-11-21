library(ggpubr)
library(ggplot2)
library(ggsci)
library(ggsignif)
library(gghalves)
library(tidyverse)


##############################
#  PPV analaysis
##############################
# Data loading for PPV
df_VitTCR = read.table('./metrics_vittcr.tsv', header=TRUE, sep='\t')
df_VitTCR$Type = factor(df_VitTCR$Type, levels=c('Original','Trainset-only','Testset-only', 'Clustered'))
df_VitTCR$Metric = factor(df_VitTCR$Metric, levels=c('PPV','AUROC','AUPR'))
head(df_VitTCR)

df_NetTCR = read.table('./metrics_nettcr.tsv', header=TRUE, sep='\t')
df_NetTCR$Type = factor(df_NetTCR$Type, levels=c('Original','Trainset-only','Testset-only', 'Clustered'))
df_NetTCR$Metric = factor(df_NetTCR$Metric, levels=c('PPV','AUROC','AUPR'))
head(df_NetTCR)

df_ERGO = read.table('./metrics_ergo.tsv', header=TRUE, sep='\t')
df_ERGO$Type = factor(df_ERGO$Type, levels=c('Original','Trainset-only','Testset-only', 'Clustered'))
df_ERGO$Metric = factor(df_ERGO$Metric, levels=c('PPV','AUROC','AUPR'))
head(df_ERGO)


# Figure configuration
p1 = ggplot(df_VitTCR, aes(x = Type, y = Value)) +
            labs(x=' ', y = "Metric value", title = 'VitTCR', )+ 
            # geom_boxplot(aes(color=Type), outlier.shape = NA)+ #show.legend = FALSE,
            geom_line(aes(group = Fold), color = 'gray') +  
            geom_violin(aes(color=Type, fill=Type), outlier.shape = NA, width = 0.5, size=1, alpha = 0.3)+ #show.legend = FALSE,
            geom_point(aes(color=Type, group=Fold), size = 1) +  
            scale_color_aaas() +
            scale_fill_aaas() +
            facet_grid( . ~ Metric) +
            geom_signif(comparisons = list(c("Original", "Trainset-only"),c("Original", "Testset-only"), c("Original", "Clustered")), map_signif_level=TRUE, step_increase=0.1) +
            theme_classic()+
            theme(plot.title = element_text(size = 16, family = "Arial", hjust = 0.5,face = "bold"),
                  plot.subtitle = element_text(size = 14, family = "Arial", hjust = 0.5,face = "bold"),
                  axis.title.x = element_text(size = 14,family = "Arial", face = "bold"),
                  axis.title.y = element_text(size = 14,family = "Arial",face = "bold"),
                  axis.text.x = element_text(size = 13, family = "Arial",face = "bold", hjust=1, angle=30),
                  axis.text.y = element_text(size = 13, family = "Arial",face = "bold"),
                  strip.text.x = element_text(size = 13, family = "Arial",face = "bold"),
                  legend.position = "None",
                  )


p2 = ggplot(df_NetTCR, aes(x = Type, y = Value)) +
            labs(x=' ', y = "Metric value", title = 'NetTCR-2.0', )+ 
            # geom_boxplot(aes(color=Type), outlier.shape = NA)+ #show.legend = FALSE,
            geom_line(aes(group = Fold), color = 'gray') +  
            geom_violin(aes(color=Type, fill=Type), outlier.shape = NA, width = 0.5, size=1, alpha = 0.3)+ #show.legend = FALSE,
            geom_point(aes(color=Type, group=Fold), size = 1) +  
            scale_color_aaas() +
            scale_fill_aaas() +
            facet_grid( . ~ Metric) +
            geom_signif(comparisons = list(c("Original", "Trainset-only"),c("Original", "Testset-only"), c("Original", "Clustered")), map_signif_level=TRUE, step_increase=0.1) +
            theme_classic()+
            theme(plot.title = element_text(size = 16, family = "Arial", hjust = 0.5,face = "bold"),
                  plot.subtitle = element_text(size = 14, family = "Arial", hjust = 0.5,face = "bold"),
                  axis.title.x = element_text(size = 14,family = "Arial", face = "bold"),
                  axis.title.y = element_text(size = 14,family = "Arial",face = "bold"),
                  axis.text.x = element_text(size = 13, family = "Arial",face = "bold", hjust=1, angle=30),
                  axis.text.y = element_text(size = 13, family = "Arial",face = "bold"),
                  strip.text.x = element_text(size = 13, family = "Arial",face = "bold"),
                  legend.position = "None",
                  )

# # df_ERGO = df_ERGO %>% filter(Type != 'Both-clustered')
p3 = ggplot(df_ERGO, aes(x = Type, y = Value)) +
            labs(x=' ', y = "Metric value", title = 'ERGO_AE', )+ 
            # geom_boxplot(aes(color=Type), outlier.shape = NA)+ #show.legend = FALSE,
            geom_line(aes(group = Fold), color = 'gray') +  
            geom_violin(aes(color=Type, fill=Type), outlier.shape = NA, width = 0.5, size=1, alpha = 0.3)+ #show.legend = FALSE,
            # facet_grid(.~Metric)+
            geom_point(aes(color=Type, group=Fold), size = 1) + 
            scale_color_aaas() +
            scale_fill_aaas() +
            facet_grid( . ~ Metric) +
            geom_signif(comparisons = list(c("Original", "Trainset-only"),c("Original", "Testset-only"), c("Original", "Clustered")), map_signif_level=TRUE, step_increase=0.1) +
            theme_classic()+
            theme(plot.title = element_text(size = 16, family = "Arial", hjust = 0.5,face = "bold"),
                  plot.subtitle = element_text(size = 14, family = "Arial", hjust = 0.5,face = "bold"),
                  axis.title.x = element_text(size = 14,family = "Arial", face = "bold"),
                  axis.title.y = element_text(size = 14,family = "Arial",face = "bold"),
                  axis.text.x = element_text(size = 13, family = "Arial",face = "bold", hjust=1, angle=30),
                  axis.text.y = element_text(size = 13, family = "Arial",face = "bold"),
                  strip.text.x = element_text(size = 13, family = "Arial",face = "bold"),
                  legend.position = "None",
                  )


df_VitTCR = df_VitTCR %>% filter(Metric == 'PPV')
df_VitTCR$Model = "VitTCR"
df_ERGO = df_ERGO %>% filter(Metric == 'PPV')
df_ERGO$Model = "ERGO_AE"
df_for_plot = rbind(df_VitTCR, df_ERGO)
p4 = ggplot(df_for_plot, aes(x = Model, y = Value)) +
            labs(x=' ', y = "Metric value", title = 'Comparison between VitTCR and ERGO_AE (PPV)', )+ 
            # geom_boxplot(aes(color=Type), outlier.shape = NA)+ #show.legend = FALSE,
            geom_line(aes(group = Fold), color = 'gray') +  , 
            geom_violin(aes(color=Type, fill=Type), outlier.shape = NA, width = 0.5, size=1, alpha = 0.3)+ #show.legend = FALSE,
            geom_point(aes(color=Type, group=Fold), size = 1) +  
            scale_color_aaas() +
            scale_fill_aaas() +
            facet_grid( . ~ Type) +
            geom_signif(comparisons = list(c("VitTCR", "ERGO_AE")), map_signif_level=TRUE, step_increase=0.1) +
            theme_classic()+
            theme(plot.title = element_text(size = 16, family = "Arial", hjust = 0.5,face = "bold"),
                  plot.subtitle = element_text(size = 14, family = "Arial", hjust = 0.5,face = "bold"),
                  axis.title.x = element_text(size = 14,family = "Arial", face = "bold"),
                  axis.title.y = element_text(size = 14,family = "Arial",face = "bold"),
                  axis.text.x = element_text(size = 13, family = "Arial",face = "bold", hjust=1, angle=30),
                  axis.text.y = element_text(size = 13, family = "Arial",face = "bold"),
                  strip.text.x = element_text(size = 13, family = "Arial",face = "bold"),
                  legend.position = "None",
                  )



##############################
#  Figure draw
##############################
# png('./violin_total_VitTCR_231119.png',height=3000, width=9000,res=450)
# ggarrange(p1,
#           labels = c(" "), align = "hv",
#           ncol=1, nrow=1, widths = c(1)
#           ) # common.legend = TRUE, legend = "top"
# dev.off()

# png('./violin_total_NetTCR_231119.png',height=3000, width=9000,res=450)
# ggarrange(p2,
#           labels = c(" "), align = "hv",
#           ncol=1, nrow=1, widths = c(1)
#           ) # common.legend = TRUE, legend = "top"
# dev.off()

# png('./violin_total_ERGO_231119.png',height=3000, width=9000,res=450)
# ggarrange(p3,
#           labels = c(" "), align = "hv",
#           ncol=1, nrow=1, widths = c(1)
#           ) # common.legend = TRUE, legend = "top"
# dev.off()

png('./vittcr_ergo_comp_231120.png',height=3000, width=9000,res=450)
ggarrange(p4,
          labels = c(" "), align = "hv",
          ncol=1, nrow=1, widths = c(1)
          ) # common.legend = TRUE, legend = "top"
dev.off()