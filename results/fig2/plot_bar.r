library(ggpubr)
library(ggplot2)
library(ggsci)
library(ggsignif)
library(ggsci)
library(tidyverse)


# Data loading
vittcr = read.table('./metrics_vittcr.tsv', sep='\t', header=TRUE)
colnames(vittcr) = c('repeat', 'epoch', 'AUROC', 'AUPR')
vittcr = vittcr[1:25, c('AUROC','AUPR')]
vittcr$Method = 'VitTCR'
vittcr$Fold = rep(1:25, times=1)

vittcr_auroc = vittcr[,c('Method','Fold','AUROC')]
colnames(vittcr_auroc) = c('Method','Fold','Metric')
vittcr_auroc$Type = 'AUROC'
vittcr_aupr = vittcr[,c('Method','Fold','AUPR')]
colnames(vittcr_aupr) = c('Method','Fold','Metric')
vittcr_aupr$Type = 'AUPR'
vittcr =  rbind(vittcr_auroc, vittcr_aupr)


nettcr = read.table('./metrics_nettcr.tsv', sep='\t', header=TRUE)
colnames(nettcr) = c('repeat', 'epoch', 'AUROC', 'AUPR')
nettcr = nettcr[1:25, c('AUROC','AUPR')]
nettcr$Method = 'NetTCR-2.0'
nettcr$Fold = rep(1:25, times=1)

nettcr_auroc = nettcr[,c('Method','Fold','AUROC')]
colnames(nettcr_auroc) = c('Method','Fold','Metric')
nettcr_auroc$Type = 'AUROC'
nettcr_aupr = nettcr[,c('Method','Fold','AUPR')]
colnames(nettcr_aupr) = c('Method','Fold','Metric')
nettcr_aupr$Type = 'AUPR'
nettcr =  rbind(nettcr_auroc, nettcr_aupr)


ergo = read.table('./metrics_ergo.tsv', sep='\t', header=TRUE)
colnames(ergo) = c('repeat', 'epoch', 'AUROC', 'AUPR')
ergo = ergo[1:25, c('AUROC','AUPR')]
ergo$Method = 'ERGO_AE'
ergo$Fold = rep(1:25, times=1)

ergo_auroc = ergo[,c('Method','Fold','AUROC')]
colnames(ergo_auroc) = c('Method','Fold','Metric')
ergo_auroc$Type = 'AUROC'
ergo_aupr = ergo[,c('Method','Fold','AUPR')]
colnames(ergo_aupr) = c('Method','Fold','Metric')
ergo_aupr$Type = 'AUPR'
ergo =  rbind(ergo_auroc, ergo_aupr)

total = rbind(vittcr,nettcr,ergo)
total$Method=factor(total$Method,level=c("VitTCR","NetTCR-2.0","ERGO_AE"))

mypal = c('#3E4A7B','#BDBDBD','#A94643')

# Plotting
p1 = ggplot(total, aes(x = Method, y = Metric)) +
    labs(x='Method', y = "Metric") +
    geom_line(aes(group = Fold), lwd = 0.2, color = 'gray') +  
    geom_violin(aes(color=Method, fill=Method), alpha=0.5, outlier.shape = NA,  width = 0.6)+ #show.legend = FALSE,
    geom_boxplot(aes(color=Method, fill=Method), alpha=0.5, outlier.shape = NA,  width = 0.6)+ #show.legend = FALSE,
    stat_summary(geom="text", fun.y=median,
                 aes(label=sprintf("%1.4f", ..y..)), # color=factor(Sample)
                 position=position_nudge(x=0.4), size=3.5) +
    # scale_color_npg() +
    # scale_fill_npg() +
    scale_color_manual(values = mypal)+
    scale_fill_manual(values = mypal)+
    facet_grid(.~Type)+
    geom_signif(comparisons = list(c("VitTCR","NetTCR-2.0"), c("VitTCR","ERGO_AE")), map_signif_level=FALSE, step_increase=0.1, method=t.test, paired = TRUE)+
    theme_classic()+
    theme(plot.title = element_text(size = 16, family = "Arial", hjust = 0.5,face = "bold"),
            plot.subtitle = element_text(size = 14, family = "Arial", hjust = 0.5,face = "bold"),
            axis.title.x = element_text(size = 14,family = "Arial", face = "bold"),
            axis.title.y = element_text(size = 14,family = "Arial",face = "bold"),
            axis.text.x = element_text(size = 13, family = "Arial",face = "bold"), # hjust = 1, angle=30
            axis.text.y = element_text(size = 13, family = "Arial",face = "bold"),
            strip.text.x = element_text(size = 13, family = "Arial",face = "bold"),
            legend.position = "None",
    )

png('./comp_merge.png',height=2000, width=4000,res=400)
ggarrange(p1,
          labels = c(" "),
          align = "hv"
      #     ncol = 1, nrow = 1
          ) #common.legend = TRUE, legend = "top", 
dev.off()