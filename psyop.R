rm(list=ls())

library(data.table)
library(lubridate)
library(scales)
library(hrbrthemes)
library(ggplot2)
library(dplyr)
library(fs)

setwd(paste(fs::path_home(),'/OneDrive/manuscripts/psyop/data',sep=""))

#### compile data for graphing
facebook = fread("./oldtonew_psyop_meta/2024-01-14-09-15-26-NZDT-search-csv-export.csv")
facebook$'Post Created' = as.POSIXct(facebook$'Post Created', tz = "NZST")
insta = fread("./oldtonew_psyop_meta/2024-01-14-09-16-36-NZDT-search-csv-export.csv")
insta$'Post Created' = as.POSIXct(insta$'Post Created' , tz = "NZST")

trends = fread("googleTrendsWorldwide.csv")
trends$Month = as.Date(paste(trends$Month,"-01",sep=""))
colnames(trends) = c("Month","Trends")
trends$Trends = trends$Trends

telegram = fread("./telegram.tsv",sep="\t",header=FALSE,quote=FALSE)
telegram$V2 = as.POSIXct(telegram$V2, origin="1970-01-01")

FOURchan = fread("./4plebs_2024.tsv",sep="\t",header=FALSE,quote=FALSE)
FOURchan$V1 = as.POSIXct(FOURchan$V1, origin="1970-01-01")

combo = data.table(Date=c(as.Date(facebook$'Post Created'),as.Date(insta$'Post Created'),as.Date(telegram$V2),as.Date(FOURchan$V1)),
                   Platform=as.factor(c(rep("Meta",(length(facebook$Likes)+length(insta$Likes))),rep("Telegram",length(telegram$V2)),rep("4chan",length(FOURchan$V1))))
                   )


colorPalette = c("#007113","#333333","#176ae2","#b30000")

my_theme <- function(){
  list(
    theme_ipsum_rc(),
    scale_color_manual(values = colorPalette),
    scale_fill_manual(values = colorPalette),
    scale_linetype_manual(values = c(2,4,1,5))
  )
}

theData <- combo %>% mutate(Date = floor_date(Date, unit = "day")) %>% group_by(Date,Platform) %>% summarize(Posts = n())
# add in trends
theData = rbind(theData,data.table(Date=trends$Month,Platform=rep("Google Trends",length(trends$Month)),Posts=trends$Trends))

theData = filter(theData,Date>=as.Date("2014-01-01"),Date<as.Date("2024-01-01"))

tmp = ggplot(data=theData,aes(x=Date,y=Posts,color=Platform)) + stat_smooth(method="loess",size=1.8,se=FALSE,span=0.15,aes(linetype=Platform))+my_theme()+theme(legend.position = "top",legend.text = element_text(size = 15))+guides(color=guide_legend(title="Platform"),linetype=guide_legend(title="Platform"))+scale_x_date(date_breaks = "6 month", date_labels =  "%b-%Y",name="")+scale_y_log10(label=comma,limits = c(1,1000))+ theme(axis.text.x = element_text(angle = 90, vjust = 0.5, hjust=1),plot.margin = margin(t = 0.2,r = 1,b = 0.2,l = 1,unit = "mm"))
tmp = tmp + geom_vline(xintercept = as.numeric(as.Date("2020-03-13")), linetype=2,colour="black",alpha=0.5)  
tmp = tmp + ggplot2::annotate(geom="label", x=as.Date("2020-03-13"), y=c(600), label=c("COVID19 US\nnational emergency"),color="black",box.padding = 0.3,angle=0)

# change point analysis via BEAST

for(platform in unique(theData$Platform)) {
  library(Rbeast)
  library(lubridate)
  library(tidyr)
  library(dplyr)
  original_df = theData %>% filter(Platform==platform) %>% select(-Platform) 
  original_df$Date = as.Date(original_df$Date)
  df = original_df %>% mutate(Date = floor_date(Date, unit = "1 month")) %>% summarize(Posts = sum(Posts))
  df = df %>% complete(Date = seq(min(Date), max(Date),by = "1 month"))
  df = df[order(df$Date),]
  out=beast(df$Posts, season='none')
  ragg::agg_png(paste("./beast_diagnostic_",platform,".png",sep=""), width = 1050, height = 700, units = "px", res = 300, scaling=0.35)
  plot(out)
  dev.off()
  
  sink(file=paste("./beast_diagnostic_",platform,".txt",sep=""))
  print(out)
  
  for(i in 1:length(unlist(out$trend["cpPr"]))) {
    cp = unlist(out$trend["cpPr"])[i]
    if(!is.na(cp) & cp>0.90) {
      point = original_df[which(original_df$Date==df[unlist(out$trend["cp"])[i],]$Date),]
      print(paste(platform,point$Date,point$Posts))
      tmp = tmp + geom_point(size=5,shape=13,aes(color=Platform,fill=Platform),data=data.table(Date=c(point$Date),Platform=c(platform),Posts=c(point$Posts)))
    }
  }
  sink(file=NULL)
}



ragg::agg_png("./psyop_timeline_2025.png", width = 1050, height = 700, units = "px", res = 300, scaling=0.35)
print(tmp)
dev.off()



### Load up BERTopic results

library(data.table)
library(stringr)
results_bert = fread("./results_MULTIPLEREPDOCS_hdbscan_psyop0.005_mcp.tsv",header=TRUE,sep="\t",quote=FALSE)
results_all_sentences = fread("./all_sentences_psyop.tsv",header=TRUE,sep="\t",quote=FALSE)
unique_sentences = fread("./uniques_psyop.tsv",header=FALSE,sep="\t",quote=FALSE)
all_posts = fread("./all_psyop_posts_combo.tsv",header=FALSE,sep="\t",quote=FALSE)
all_posts$V1 = as.Date(as.POSIXct(all_posts$V1, origin="1970-01-01", tz="UTC"))
hdbscan_results = fread("./hdbscan_psyop0.005_mcp.tsv",header=TRUE,sep="\t",quote=FALSE)


#index is zero based for python, populate all sentences with topics and dates
results_all_sentences$Date = all_posts[results_all_sentences$index+1]$V1
results_all_sentences$Topic = hdbscan_results[match(results_all_sentences$sentence,unique_sentences$V1)]
results_all_sentences$Platform = all_posts[results_all_sentences$index+1]$V3
results_all_sentences$TopicName = results_bert[match(results_all_sentences$Topic,results_bert$Topic)]$Name


#reconstruct audience and engagement data
facebook = fread("./oldtonew_psyop_meta/2024-01-14-09-15-26-NZDT-search-csv-export.csv",na.strings = c("", "NA","<NA>","N/A"))
facebook$'Post Created' = as.POSIXct(facebook$'Post Created', tz = "NZST")
insta = fread("./oldtonew_psyop_meta/2024-01-14-09-16-36-NZDT-search-csv-export.csv",na.strings = c("", "NA","<NA>","N/A"))
insta$'Post Created' = as.POSIXct(insta$'Post Created' , tz = "NZST")

audience = as.numeric(pmax(
  c(facebook$`Likes at Posting`,rep(NA,length(insta$Account))),
  c(facebook$`Followers at Posting`,insta$`Followers at Posting`),
  na.rm = TRUE))
totalInteractions = as.numeric(c(facebook$`Total Interactions`,insta$`Total Interactions`))

all_posts$Audience = c(rep(NA,length(FOURchan$V1)),audience,rep(NA,length(telegram$V1)))
all_posts$totalInteractions = c(rep(NA,length(FOURchan$V1)),totalInteractions,rep(NA,length(telegram$V1)))

results_all_sentences$Audience = all_posts[results_all_sentences$index+1]$Audience
results_all_sentences$totalInteractions = all_posts[results_all_sentences$index+1]$totalInteractions

meta_proportions_table = data.table(hdbscan=c(),audience=c(),engagement=c())
english_classified = filter(results_all_sentences,Topic!="-1",Topic!="1",Topic!="3",Topic!="2")
nonpsyop_artist = filter(english_classified,Topic!="54")


### draw up proportions table and create thumbnail graphs
library(lubridate)
library(dplyr)
library(ggplot2)
library(hrbrthemes)
library(scales)
library(data.table)
customPalette = c("#007113","#176ae2","#b30000")
my_theme <- function(){
  list(
    theme_ipsum_rc(),
    scale_color_manual(values = customPalette),
    scale_fill_manual(values = customPalette),
    scale_linetype_manual(values = c(2,1,5))
  )
}

topics = unique(nonpsyop_artist$Topic)
for(cluster in topics) {
  cluster_sentences = filter(results_all_sentences,Topic==cluster)
  
  cluster_sentences$Platform <-
    plyr::revalue(
      cluster_sentences$Platform,
      c(
        "4chan" = "4chan",
        "Facebook" = "Meta",
        "Instagram" = "Meta",
        "Telegram" = "Telegram"
      )) %>% factor()

  theGraphData <- cluster_sentences %>% group_by(Date,Platform) %>% summarize(Posts = n())

  tmp = ggplot(data=theGraphData,aes(x=Date,y=Posts,color=Platform)) +
    stat_smooth(method="loess",size=1.2,se=FALSE,span=0.2,aes(linetype=Platform))+
    my_theme()+theme(legend.position = "none")+
    scale_y_log10(labels = label_number(accuracy = 1))+
    scale_x_date(date_breaks = "3 year", date_labels =  "%Y",name="Day")+
    geom_vline(xintercept = as.numeric(as.Date("2020-03-01")), linetype=2,colour="black",alpha=0.5)+
    theme(legend.position = "none",axis.text.x=element_text(size=13),axis.title.y = element_blank(),axis.text.y = element_text(size=12),axis.title.x= element_blank(),plot.margin = margin(t = 0.1,r = 0.1,b = 0.1,l = 0.1,unit = "mm"))
  
  
  for(platform in unique(theGraphData$Platform)) {
    library(Rbeast)
    library(lubridate)
    library(tidyr)
    library(dplyr)
    original_df = theGraphData %>% filter(Platform==platform) %>% select(-Platform) 
    original_df$Date = as.Date(original_df$Date)
    df = original_df %>% mutate(Date = floor_date(Date, unit = "1 month")) %>% summarize(Posts = sum(Posts))
    df = df %>% complete(Date = seq(min(Date), max(Date),by = "1 month"))
    df = df[order(df$Date),]
    out=beast(df$Posts, season='none',hasOutlier = TRUE)
    sink(file=paste("./thumbnails/thumbnails_beast_diagnostic_",cluster,"_",platform,".txt",sep=""))
    print(out)
    
    for(i in 1:length(unlist(out$trend["cpPr"]))) {
      cp = unlist(out$trend["cpPr"])[i]
      if(!is.na(cp) & cp>0.90) {
        point = original_df[which(original_df$Date==df[unlist(out$trend["cp"])[i],]$Date),]
        print(paste(platform,point$Date,point$Posts))
        tmp = tmp + geom_point(size=5,shape=13,aes(color=Platform,fill=Platform),data=data.table(Date=c(point$Date),Platform=c(platform),Posts=c(point$Posts)))
      }
    }
    sink(file=NULL)
  }
  
  ragg::agg_png(paste("./thumbnails/timeseries_",cluster,".png",sep=""), width = 176, height = 99, units = "px", res = 60, scaling=1)
  print(tmp)
  dev.off()

  meta_tmp = filter(cluster_sentences,Platform=='Meta')
  topic_percentOfReach = sum(meta_tmp$Audience,na.rm=TRUE)/sum(filter(results_all_sentences,Platform=='Facebook' | Platform=='Instagram')$Audience,na.rm=TRUE)
  topic_percentOfSocialInteractions = sum(meta_tmp$totalInteractions,na.rm=TRUE)/sum(filter(results_all_sentences,Platform=='Facebook' | Platform=='Instagram')$totalInteractions,na.rm=TRUE)
  meta_proportions_table = rbind(meta_proportions_table,data.table(hdbscan=c(cluster),audience=c(topic_percentOfReach),engagement=c(topic_percentOfSocialInteractions)))
  
  library(RColorBrewer)
  # get rid of green
  set1mod = brewer.pal(n = 6, name = "Set1")[4:5]
  my_bartheme <- function(){
    list(
      theme_ipsum_rc(),
      scale_fill_manual(values =  set1mod),
      scale_color_manual(values = set1mod)
    )
  }

  bardata = data.table(type=c("audience","social media engagements"),percent=round(c(topic_percentOfReach,topic_percentOfSocialInteractions)*100,1))
  tmp_barplot = ggplot(data=bardata,aes(x=type,y=percent,fill=type,color=type)) +
    geom_col(show.legend = FALSE)+
    my_bartheme()+coord_flip()+geom_text(aes(label = paste(percent,"%",sep="")),size = 13,hjust = -0.1, show.legend=FALSE)+
    scale_y_continuous(labels = label_number(accuracy = 1),limits = c(0,20),breaks=c(5, 15))+
    theme(legend.position = "none",axis.title.x = element_blank(),axis.text.x=element_text(size=13),axis.title.y = element_blank(),axis.text.y = element_blank(),plot.margin = margin(t = 0.1,r = 0.1,b = 0.2,l = 0.1,unit = "mm"))
  ragg::agg_png(paste("./thumbnails/barplot_",cluster,".png",sep=""), width = 176, height = 99, units = "px", res = 50, scaling=1)
  print(tmp_barplot)
  dev.off()
  
  
}


library(grid)
library(gridExtra) 
bardata = data.table(type=c("audience","social media engagements"),percent=round(c(topic_percentOfReach,topic_percentOfSocialInteractions)*100,2),levels=levels(as.factor(c("social media engagements","audience"))))

tmp_barplot = ggplot(data=bardata,aes(x=type,y=percent,fill=type,color=type)) +
  geom_col(show.legend = FALSE)+
  my_bartheme()+coord_flip()+geom_text(aes(label = paste(percent,"%",sep="")),size = 13,hjust = -0.1, show.legend=FALSE)+ylim(0,13)+
  theme(legend.position = "none",axis.title.x = element_blank(),axis.text.x=element_blank(),axis.title.y = element_blank(),axis.text.y = element_blank(),plot.margin = margin(t = 0.1,r = 0.1,b = 0.1,l = 0.1,unit = "mm"))
tmp_barplotLEGEND = tmp_barplot + theme(legend.position = "right", legend.title = element_blank(), text = element_text(size=15))+geom_col(show.legend = TRUE)+scale_fill_manual(breaks=c("social media engagements","audience"),values =  rev(set1mod))+scale_color_manual(breaks=c("social media engagements","audience"),values =  rev(set1mod))
legend <- cowplot::get_legend(tmp_barplotLEGEND)
tmplabel = grid.newpage()
ragg::agg_png(paste("./thumbnails/barplot_legend.png"), width = 176, height = 30, units = "px", res = 60, scaling=1)
grid.draw(legend)
dev.off()


library(grid)
library(gridExtra) 
tmp_timeseriesLEGEND = tmp + theme(legend.position = "top",legend.title=element_blank(),legend.text = element_text(family='serif',size=17),legend.key.spacing.x = unit(5, "mm"),legend.margin=margin(0, 18, 0, 0))
legend <- cowplot::get_plot_component(tmp_timeseriesLEGEND, 'guide-box-top', return_all = TRUE)
tmplabel = grid.newpage()
ragg::agg_png(paste("./thumbnails/timeseries_legend.png"), width = 176, height = 30, units = "px", res = 50, scaling=1)
grid.draw(legend)
dev.off()




# # export results and images to html
library(xtable)
tmp_results_bert = results_bert
tmp_results_bert$Timeline = paste("<img src=\"/home/justin/OneDrive/manuscripts/psyop/data/thumbnails/timeseries_",results_bert$Topic,".png\" /img>",sep="")
tmp_results_bert$Socials = paste("<img src=\"/home/justin/OneDrive/manuscripts/psyop/data/thumbnails/barplot_",results_bert$Topic,".png\" /img>",sep="")
tmp_html = gsub(pattern="/img&gt;", replacement="/>",gsub(pattern="&lt;img",replacement="<img",print(xtable(tmp_results_bert),type="html")))
cat(tmp_html,file="./htmlTables/htmlTable_standard.html")


# export appendix results and images to html
library(xtable)
tmp_results_bert = results_bert
tmp_results_bert = tmp_results_bert[,!c("Name")]
tmp_results_bert$Timeline = paste("<img src=\"",getwd(),"/thumbnails/timeseries_",results_bert$Topic,".png\" /img>",sep="")
tmp_results_bert$Socials = paste("<img src=\"",getwd(),"/thumbnails/barplot_",results_bert$Topic,".png\" /img>",sep="")
# remove noise category
tmp_results_bert = tmp_results_bert[-(which(tmp_results_bert$Topic==-1)),]
tmp_results_bert = tmp_results_bert[-(which(tmp_results_bert$Topic==1)),]
tmp_results_bert = tmp_results_bert[-(which(tmp_results_bert$Topic==2)),]
tmp_results_bert = tmp_results_bert[-(which(tmp_results_bert$Topic==3)),]
tmp_results_bert = tmp_results_bert[-(which(tmp_results_bert$Topic==54)),]

tmp_results_bert$Topic = paste("Topic: ", tmp_results_bert$Topic,";<br>Count: ",tmp_results_bert$Count,";<br><br>",tmp_results_bert$Representation,sep="")
tmp_results_bert = within(tmp_results_bert,rm("Count", "Representation"))
tmp_results_bert$Timeline = paste(tmp_results_bert$Timeline,"<br>",tmp_results_bert$Socials,sep="")
tmp_results_bert = within(tmp_results_bert,rm("Socials"))

tmp_html = gsub(pattern="/img&gt;", replacement="/>",gsub(pattern="&lt;img",replacement="<img",print(xtable(tmp_results_bert), include.rownames=FALSE,type="html")))
replacementString = paste("<th> Timeline &<br>Socials<br><img src='",getwd(),"/thumbnails/timeseries_legend.png' /img>","<br>","<img src='",getwd(),"/thumbnails/barplot_legend.png' /img></th>","</th>",sep="")
tmp_html = gsub(pattern="<th> Timeline </th>", replacement=replacementString,tmp_html)
tmp_html = gsub(pattern="&lt;br&gt;", replacement="<br>",tmp_html)


cat(tmp_html,file="./htmlTables/htmlTable_multirep.html")

#investigate top Meta topics
audiences = meta_proportions_table[order(meta_proportions_table$audience,decreasing=TRUE)][1:50]
engagements = meta_proportions_table[order(meta_proportions_table$engagement,decreasing=TRUE)][1:50]
both = filter(meta_proportions_table,hdbscan %in% audiences | hdbscan %in% engagements$hdbscan)
both = both[order(both$engagement,decreasing = TRUE)]

library(xtable)
abridged_results_bert = results_bert %>% filter(Topic %in% both$hdbscan)
abridged_results_bert = abridged_results_bert[match(both$hdbscan, abridged_results_bert$Topic),]
abridged_results_bert <- subset(abridged_results_bert, select = c("Topic","Name", "Representation"))
abridged_results_bert$Timeline = paste("<img src=\"/home/justin/OneDrive/manuscripts/psyop/data/thumbnails/timeseries_",abridged_results_bert$Topic,".png\" /img>",sep="")
abridged_results_bert$Socials = paste("<img src=\"/home/justin/OneDrive/manuscripts/psyop/data/thumbnails/barplot_",abridged_results_bert$Topic,".png\" /img>",sep="")
abridged_results_bert = abridged_results_bert[,!c("Topic")]
abridged_results_html = gsub(pattern="/img&gt;", replacement="/>",gsub(pattern="&lt;img",replacement="<img",print(xtable(abridged_results_bert),type="html")))
cat(abridged_results_html,file="./htmlTables/htmlTable_abridged.html")