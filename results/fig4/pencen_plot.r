# @Author:    Mengnan Jiang
# @Created:   2023/08/14


library(ggpubr)
library(ggplot2)
library(ggsci)
library(ggsignif)
library(tidyverse)
library(plyr)

colors=c('#9F9CC6','#66519F')

# Dataloading
# clustered
df_train = read.table('filtered_epitope_percentile_train.tsv', header=TRUE, sep='\t')
df_train$Database = 'Training set'

df_test = read.table('filtered_epitope_percentile_test.tsv', header=TRUE, sep='\t')
df_test$Database = 'Test set'

df_total_clustered = rbind(df_train, df_test)
df_total_clustered$Type = 'After'

# original
df_train = read.table('original_epitope_percentile_train.tsv', header=TRUE, sep='\t')
df_train$Database = 'Training set'

df_test = read.table('original_epitope_percentile_test.tsv', header=TRUE, sep='\t')
df_test$Database = 'Test set'

df_total_original = rbind(df_train, df_test)
df_total_original$Type = 'Before'

df_total = rbind(df_total_clustered, df_total_original)
df_total$Type=factor(df_total$Type, level=c("Before", "After"))


# Plotting
colors_for_select=pal_npg("nrc", alpha = 1)(3)
mypal=colors_for_select

p = ggplot(df_total, aes(x = Type, y = percentile)) +
    labs(x=' ', y = "Percentile (%)") +
    geom_violin(aes(color=Type, fill=Type), alpha=0.5, outlier.shape = NA,  width = 0.6)+ #show.legend = FALSE,
    geom_boxplot(aes(color=Type, fill=Type), alpha=0.5, outlier.shape = NA,  width = 0.6)+ #show.legend = FALSE,
    stat_summary(geom="text", fun.y=median,
                aes(label=sprintf("%1.4f", ..y..)), # color=factor(Sample)
                position=position_nudge(x=0.4), size=3.5) +
    # scale_color_aaas() +
    # scale_fill_aaas() +
    facet_grid(.~Database)+
    scale_color_manual(values = colors)+
    scale_fill_manual(values = colors)+
    theme_classic()+
    theme(plot.title = element_text(size = 16, family = "Arial", hjust = 0.5,face = "bold"),
            plot.subtitle = element_text(size = 14, family = "Arial", hjust = 0.5,face = "bold"),
            axis.title.x = element_text(size = 14,family = "Arial", face = "bold"),
            axis.title.y = element_text(size = 14,family = "Arial",face = "bold"),
            axis.text.x = element_text(size = 13, family = "Arial",face = "bold"), #hjust = 1, angle=30
            axis.text.y = element_text(size = 13, family = "Arial",face = "bold"),
            strip.text.x = element_text(size = 13, family = "Arial",face = "bold"),
            legend.position = "None",
    )
png('percentile_compare.png', height=3000, width=6000,res=600)
ggarrange(p,
          align = "hv",
          labels = c(" "),
          ncol = 1, nrow = 1
          ) #common.legend = TRUE, legend = "top", 
dev.off()