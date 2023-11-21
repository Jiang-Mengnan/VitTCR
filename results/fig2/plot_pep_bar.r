library(ggpubr)
library(ggplot2)
library(ggsci)
library(ggsignif)
library(ggsci)
library(tidyverse)


total = read.table('./pepbased.tsv', sep='\t', header=TRUE)
total$method=factor(total$method,level=c("VitTCR","NetTCR-2.0","ERGO_AE"))

# Select the counts of color
mypal = c('#3E4A7B','#BDBDBD','#A94643')
# samples = unique(total$method)
# colors_for_select=pal_npg("nrc", alpha = 1)(10)
# if (length(samples) > 10){
#   mypal=colorRampPalette(colors_for_select)(length(samples))
# } else{
#   mypal=colors_for_select
# }


# Plotting
data1 = total %>% filter(type == "unseen") # %>% filter(metric_type == "AUROC") 
p1 = ggplot(data1, aes(x = method, y = metric)) +
    labs(x='Method', y = "New epitopes") +
    # geom_line(aes(group = pep), lwd = 0.2, color = 'gray') + 
    geom_violin(aes(color=method, fill=method), alpha=0.5, outlier.shape = NA,  width = 0.6)+ #show.legend = FALSE,
    geom_boxplot(aes(color=method, fill=method), alpha=0.5, outlier.shape = NA,  width = 0.6)+ #show.legend = FALSE,
    stat_summary(geom="text", fun.y=median,
                 aes(label=sprintf("%1.4f", ..y..)), # color=factor(Sample)
                 position=position_nudge(x=0.4), size=3.5) +
    # scale_color_npg() +
    # scale_fill_npg() +
    scale_color_manual(values = mypal)+
    scale_fill_manual(values = mypal)+
    facet_grid(.~metric_type)+
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
png('./pep_based_unseen.png',height=2000, width=4000,res=400)
ggarrange(p1,
          labels = c(" "),
          align = "hv"
          ) #ncol = 1, nrow = 1, common.legend = TRUE, legend = "top", 
dev.off()




data2 = total %>% filter(type == "seen") # %>% filter(metric_type == "AUROC") 
p1 = ggplot(data2, aes(x = method, y = metric)) +
    labs(x='Method', y = "Known epitopes") +
    # geom_line(aes(group = pep), lwd = 0.2, color = 'gray') +  
    geom_violin(aes(color=method, fill=method), alpha=0.5, outlier.shape = NA,  width = 0.6)+ #show.legend = FALSE,
    geom_boxplot(aes(color=method, fill=method), alpha=0.5, outlier.shape = NA,  width = 0.6)+ #show.legend = FALSE,
    stat_summary(geom="text", fun.y=median,
                 aes(label=sprintf("%1.4f", ..y..)), # color=factor(Sample)
                 position=position_nudge(x=0.4), size=3.5) +
    # scale_color_npg() +
    # scale_fill_npg() +
    scale_color_manual(values = mypal)+
    scale_fill_manual(values = mypal)+
    facet_grid(.~metric_type)+
    geom_signif(comparisons = list(c("VitTCR","NetTCR-2.0"), c("VitTCR","ERGO_AE")), map_signif_level=FALSE, step_increase=0.1, method=t.test, paired = TRUE)+
    theme_classic()+
    theme(plot.title = element_text(size = 16, family = "Arial", hjust = 0.5,face = "bold"),
            plot.subtitle = element_text(size = 14, family = "Arial", hjust = 0.5,face = "bold"),
            axis.title.x = element_text(size = 14,family = "Arial", face = "bold"),
            axis.title.y = element_text(size = 14,family = "Arial",face = "bold"),
            axis.text.x = element_text(size = 13, family = "Arial",face = "bold"),
            axis.text.y = element_text(size = 13, family = "Arial",face = "bold"),
            strip.text.x = element_text(size = 13, family = "Arial",face = "bold"),
            legend.position = "None",
    )
png('./pep_based_seen.png',height=2000, width=4000,res=400)
ggarrange(p1,
          labels = c(" "),
          align = "hv"
          ) 
dev.off()