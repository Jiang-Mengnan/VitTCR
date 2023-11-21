library(ggpubr)
library(ggplot2)
library(ggsci)
library(ggsignif)
library(gghalves)
library(tidyverse)


colors=c('#8491B4FF','#00A087FF')


# Data loading
cols_remain=c('val_loss_min_index_PR_test','val_loss_min_index_AUC_test')

df_original = read.table('~/Project_TCR_Antigen/2.PBWM/0.Methods/VitTCR_atch_M_d1_230317/metrics_orig.tsv', header=TRUE, sep='\t')
df_original = df_original[1:25, colnames(df_original)%in%cols_remain]
df_original$Method='Original'
df_original$Id = rep(1:25, times=1)

df_attention = read.table('~/Project_TCR_Antigen/2.PBWM/0.Methods/VitTCR_atch_M_d1_230317_pbwm/metrics_pbwm.tsv', header=TRUE, sep='\t')
df_attention = df_attention[1:25, colnames(df_attention)%in%cols_remain]
df_attention$Method='PBWM'
df_attention$Id = rep(1:25, times=1)

df_total=rbind(df_original,df_attention)
df_for_plotting=df_total

########### AUROC ###########
summ_total = df_for_plotting %>%
    group_by(Method) %>%
    summarise(
    mean = mean(val_loss_min_index_AUC_test),
    sd = sd(val_loss_min_index_AUC_test),
    n = n()
  ) %>% 
  mutate(se = sd/sqrt(n),
         Method = factor(Method, levels = c('Original', 'PBWM')))
summ_total$Id = c(1,1)
summ_total$Method = factor(summ_total$Method, levels=c('Original', 'PBWM'))
summ_total
df_Lefthalf = df_for_plotting %>% filter(Method == "Original")
df_Righthalf = df_for_plotting %>% filter(Method == "PBWM")
summ_df_Lefthalf = summ_total %>% filter(Method == "Original")
summ_df_Righthalf = summ_total %>% filter(Method == "PBWM")

p1 = ggplot(df_for_plotting, aes(x=Method, y=val_loss_min_index_AUC_test, fill=Method))+
    labs(x=' ', y = "AUROC (Independent Validation)")+ #title="VitTCR",
    scale_x_discrete(limits=c("Original","PBWM"))+
    # geom_line(aes(group=Id), color="gray" ,position=position_dodge(0)) +
    geom_point(aes(color=Method, group=Id), size = 1.5, position = position_dodge(0))+
    geom_half_violin(aes(color = Method), data=df_Righthalf, side = 'r',
                    position = position_nudge(x = .10, y = 0))+
    geom_half_violin(aes(color = Method), data = df_Lefthalf, side = 'l',
                    position = position_nudge(x = -0.10, y = 0))+
    geom_boxplot(data=df_Righthalf,
                aes(x=Method, y=val_loss_min_index_AUC_test, fill=Method),
                outlier.shape = NA,
                width=.05,
                color="black",
                position = position_nudge(x = 0.15, y = 0))+
    geom_boxplot(data = df_Lefthalf,
                aes(x=Method, y=val_loss_min_index_AUC_test, fill=Method),
                outlier.shape = NA,
                width=.05,
                color="black",
                position=position_nudge(x = -0.15, y = 0))+
    geom_errorbar(data = summ_df_Righthalf,
                  aes(x=Method, y=mean, group=Method, colour=Method, ymin = mean-sd, ymax = mean+sd),
                  width=0.1,size=2,
                  position=position_nudge(x = -0.2, y = 0))+
    geom_point(data=summ_df_Righthalf,
               aes(x=Method, y=mean, group=Method, color=Method),
               size=4,
               position = position_nudge(x = -0.2,y = 0))+
    geom_point(data=summ_df_Righthalf,
               aes(x=Method,y = mean, group=Method),
               color="black",
               size=2,
               position=position_nudge(x = -0.2,y = 0))+
    geom_errorbar(data=summ_df_Lefthalf,
                  aes(x=Method, y=mean, group=Method, colour=Method, ymin=mean-sd, ymax=mean+sd),
                  width=0.1,size=2,
                  position=position_nudge(x=0.2, y=0))+
    geom_point(data=summ_df_Lefthalf,
               aes(x=Method,y = mean, group=Method, color=Method),
               size = 4,
               position = position_nudge(x = 0.2,y = 0))+
    geom_point(data=summ_df_Lefthalf,
               aes(x=Method,y = mean, group=Method),
               color = "black",
               size = 2,
               position = position_nudge(x = 0.2,y = 0))+
    scale_fill_manual(values = colors)+
    scale_color_manual(values = colors)+
    geom_signif(comparisons = list(c("Original", "PBWM")), map_signif_level=TRUE, method=t.test, paired = TRUE) +
    theme_classic()+
    theme(plot.title = element_text(size = 16, family = "Arial",face = "bold", hjust = 0.5),
          plot.subtitle = element_text(size = 14, family = "Arial",face = "bold", hjust = 0.5),
          axis.title.x = element_text(size = 14,family = "Arial",face = "bold"),
          axis.title.y = element_text(size = 14,family = "Arial",face = "bold"),
          axis.text.x = element_text(size = 13, family = "Arial",face = "bold"),
          axis.text.y = element_text(size = 13, family = "Arial",face = "bold"),
          strip.text.x = element_text(size = 13, family = "Arial",face = "bold"),
          legend.position = "None",
          )


########### AUPR ###########
summ_total = df_for_plotting %>%
    group_by(Method) %>%
    summarise(
    mean = mean(val_loss_min_index_PR_test),
    sd = sd(val_loss_min_index_PR_test),
    n = n()
  ) %>% 
  mutate(se = sd/sqrt(n),
         Method = factor(Method, levels = c('Original', 'PBWM')))
summ_total$Id = c(1,1)
summ_total$Method = factor(summ_total$Method, levels=c('Original', 'PBWM'))
summ_total
df_Lefthalf = df_for_plotting %>% filter(Method == "Original")
df_Righthalf = df_for_plotting %>% filter(Method == "PBWM")
summ_df_Lefthalf = summ_total %>% filter(Method == "Original")
summ_df_Righthalf = summ_total %>% filter(Method == "PBWM")

p2 = ggplot(df_for_plotting, aes(x=Method, y=val_loss_min_index_PR_test, fill=Method))+
    labs(x=' ', y = "AUPR (Independent Validation)")+ #title="VitTCR",
    scale_x_discrete(limits=c("Original","PBWM"))+
    # geom_line(aes(group=Id), color="gray" ,position=position_dodge(0)) +
    geom_point(aes(color=Method, group=Id), size = 1.5, position = position_dodge(0))+
    geom_half_violin(aes(color = Method), data=df_Righthalf, side = 'r',
                    position = position_nudge(x = .10, y = 0))+
    geom_half_violin(aes(color = Method), data = df_Lefthalf, side = 'l',
                    position = position_nudge(x = -0.10, y = 0))+
    geom_boxplot(data=df_Righthalf,
                aes(x=Method, y=val_loss_min_index_PR_test, fill=Method),
                outlier.shape = NA,
                width=.05,
                color="black",
                position = position_nudge(x = 0.15, y = 0))+
    geom_boxplot(data = df_Lefthalf,
                aes(x=Method, y=val_loss_min_index_PR_test, fill=Method),
                outlier.shape = NA,
                width=.05,
                color="black",
                position=position_nudge(x = -0.15, y = 0))+
    geom_errorbar(data = summ_df_Righthalf,
                  aes(x=Method, y=mean, group=Method, colour=Method, ymin = mean-sd, ymax = mean+sd),
                  width=0.1,size=2,
                  position=position_nudge(x = -0.2, y = 0))+
    geom_point(data=summ_df_Righthalf,
               aes(x=Method, y=mean, group=Method, color=Method),
               size=4,
               position = position_nudge(x = -0.2,y = 0))+
    geom_point(data=summ_df_Righthalf,
               aes(x=Method,y = mean, group=Method),
               color="black",
               size=2,
               position=position_nudge(x = -0.2,y = 0))+
    geom_errorbar(data=summ_df_Lefthalf,
                  aes(x=Method, y=mean, group=Method, colour=Method, ymin=mean-sd, ymax=mean+sd),
                  width=0.1,size=2,
                  position=position_nudge(x=0.2, y=0))+
    geom_point(data=summ_df_Lefthalf,
               aes(x=Method,y = mean, group=Method, color=Method),
               size = 4,
               position = position_nudge(x = 0.2,y = 0))+
    geom_point(data=summ_df_Lefthalf,
               aes(x=Method,y = mean, group=Method),
               color = "black",
               size = 2,
               position = position_nudge(x = 0.2,y = 0))+
    scale_fill_manual(values = colors)+
    scale_color_manual(values = colors)+
    geom_signif(comparisons = list(c("Original", "PBWM")), map_signif_level=TRUE, method=t.test, paired = TRUE) +
    theme_classic()+
    theme(plot.title = element_text(size = 16, family = "Arial",face = "bold", hjust = 0.5),
          plot.subtitle = element_text(size = 14, family = "Arial",face = "bold", hjust = 0.5),
          axis.title.x = element_text(size = 14,family = "Arial",face = "bold"),
          axis.title.y = element_text(size = 14,family = "Arial",face = "bold"),
          axis.text.x = element_text(size = 13, family = "Arial",face = "bold"),
          axis.text.y = element_text(size = 13, family = "Arial",face = "bold"),
          strip.text.x = element_text(size = 13, family = "Arial",face = "bold"),
          legend.position = "None",
          )






##############################
#  Figure draw
##############################
figpath='comp.png'
png(figpath,height=1500, width=3000,res=300)
ggarrange(p1,p2,
          # labels = c("a","b"),
          align = "hv",
          ncol = 2, nrow = 1
          ) #common.legend = FALSE,legend = "top", 
dev.off()
