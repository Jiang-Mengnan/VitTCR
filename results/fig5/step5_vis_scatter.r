library(ggpubr)
library(ggplot2)
library(ggsci)
library(ggsignif)
library(gghalves)
library(tidyverse)


# Data loading
Donor1 = read.table('../result_of_VitTCR/1.Prediction/Average_Donor_1_HLA_A0201.csv', header=TRUE, sep=',')
Donor1$donor='Donor 1'

Donor2 = read.table('../result_of_VitTCR/1.Prediction/Average_Donor_2_HLA_A0201.csv', header=TRUE, sep=',')
Donor2$donor='Donor 2'

figpath='../result_of_VitTCR/Visualization'
if (!dir.exists(figpath)) {
    dir.create(figpath)
}

p_donor1 = ggplot(Donor1, aes(x = prob_1, y = percent)) +
            labs(x='Predicted Probability', y = "Clone Fraction", title = 'Donor 1')+ 
            geom_point(size = 1.2,color='#3E4A7B') +  
            # geom_smooth(method = "lm", color = "black", fill = "lightgray")+
            stat_cor(data = Donor1, method = "spearman")+
            ylim(0, 0.0005)+
            scale_color_jco() +
            scale_fill_jco() +
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

p_donor2 = ggplot(Donor2, aes(x = prob_1, y = percent)) +
            labs(x='Predicted Probability', y = "Clone Fraction", title = 'Donor 2')+ 
            geom_point(size = 1.2,color='#3E4A7B') +  
            # geom_smooth(method = "lm", color = "black", fill = "lightgray")+
            stat_cor(data = Donor2, method = "spearman")+
            ylim(0, 0.0005)+
            scale_color_jco() +
            scale_fill_jco() +
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

png(sprintf('%s/Prob_CloneFrac_Scatter.png',figpath),height=1800, width=3600,res=300)
figure = ggarrange(p_donor1, p_donor2,
                    align = "hv",
                    labels = c("a"), font.label = list(size = 24, face = "bold"),
                    ncol = 2, 
                    nrow = 1          
                    ) #common.legend = TRUE, legend = "top", 
figure = annotate_figure(figure, 
                         top=text_grob("VitTCR", color = "#3E4A7B", face = "bold", size=14)
                         #  fig.lab = "VitTCR", fig.lab.face = "bold", fig.lab.size=14,
                         ) #top = text_grob(sprintf("%s",sample), face = "bold", size=14),
figure
dev.off()
