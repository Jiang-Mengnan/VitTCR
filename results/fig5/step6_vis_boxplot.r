library(ggpubr)
library(ggplot2)
library(ggsci)
library(ggsignif)
library(gghalves)
library(tidyverse)


VitTCR = read.table('../result_of_VitTCR_atch_M_d1_230317/3.Summary/Summary_VitTCR_A0201.tsv', header=TRUE, sep='\t')
VitTCR$Method = 'VitTCR'

NetTCR2.0 = read.table('../result_of_NetTCR2.0/3.Summary/Summary_NetTCR2.0_A0201.tsv', header=TRUE, sep='\t')
NetTCR2.0$Method = 'NetTCR-2.0'

ERGO_AE = read.table('../result_of_ERGO_AE/3.Summary/Summary_ERGO_AE_A0201.tsv', header=TRUE, sep='\t')
ERGO_AE$Method = 'ERGO_AE'

All = rbind(VitTCR,NetTCR2.0, ERGO_AE)
All$Method=factor(All$Method,level=c("VitTCR","NetTCR-2.0","ERGO_AE"))

All$Donor[which(All$Donor =='1')] = 'Donor 1'
All$Donor[which(All$Donor =='2')] = 'Donor 2'

colors=c('#A94643','#BDBDBD','#3E4A7B')

figpath='../comparison'
if (!dir.exists(figpath)) {
    dir.create(figpath)
}

p_All = ggplot(All, aes(x = Method, y = spearmanr)) +
            labs(x='Methods', y = "Spearman Coefficients", title = ' ')+ 
            stat_boxplot(geom="errorbar",width=0.15,size=0.9,position=position_dodge(0.7))+
            # geom_boxplot(aes(fill=Method), alpha=0.7, outlier.shape = NA, width=0.5)+
            geom_boxplot(aes(fill=Method), outlier.shape = NA,size=0.5, width=0.5)+ #show.legend = FALSE,
            stat_summary(geom="text", fun.y=median,
                        aes(label=sprintf("%1.4f", ..y..)), # color=factor(Sample)
                        position=position_nudge(x=0.4), size=3.5) +
            # geom_boxplot(aes(fill=Method), outlier.shape = NA,size=0.5, width=0.5)+ #show.legend = FALSE,
            # scale_color_jco() +
            # scale_fill_jco() +
            # scale_color_npg() +
            # scale_fill_npg() +
            # scale_color_aaas()+
            # scale_fill_aaas()+
            scale_fill_manual(values = colors)+
            scale_color_manual(values = colors)+
            theme_classic()+
            facet_grid( . ~ Donor)+
            geom_signif(test = wilcox.test, 
                        test.args = c("greater"), 
                        comparisons = list(c("VitTCR", "NetTCR-2.0"), c("VitTCR", "ERGO_AE")), 
                        map_signif_level=TRUE, 
                        step_increase=0.06) +
            theme(plot.title = element_text(size = 16, family = "Arial", hjust = 0.5,face = "bold"),
                  plot.subtitle = element_text(size = 14, family = "Arial", hjust = 0.5,face = "bold"),
                  axis.title.x = element_text(size = 14,family = "Arial", face = "bold"),
                  axis.title.y = element_text(size = 14,family = "Arial",face = "bold"),
                  axis.text.x = element_text(size = 13, family = "Arial",face = "bold"),
                  axis.text.y = element_text(size = 13, family = "Arial",face = "bold"),
                  strip.text.x = element_text(size = 13, family = "Arial",face = "bold"),
                  legend.position = "None",
                  )


png(sprintf('%s/Coefficient_All_Boxplot.png',figpath),height=1800, width=3600,res=300)
figure = ggarrange(p_All,
                    align = "hv",
                    ncol = 1, 
                    nrow = 1          
                    ) #common.legend = TRUE, legend = "top", 
figure
dev.off()