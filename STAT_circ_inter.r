#module load r; R
library(Matrix)
library(qqman)
library(ggplot2) # Data visualization
library(data.table)
library(xgboost)
library(caret)
library(qvalue)
library(circlize)
library(car)
library(EnvStats)
require(sparseLDA)
require(sda)
require(gridExtra)
A=read.csv('OAI_20190621.csv',check.names = FALSE)
y=A[,1]
X=A[,-1]
p=dim(X)[2]
#boxcox(X[,3], objective.name = "Shapiro-Wilk")
modality=unlist(lapply(strsplit(colnames(X),'\\+'),'[',1))
gsub1<-function(i,vec,vec1){return(gsub(vec[i],'',vec1[i]))}
modality2=unlist(lapply(1:length(modality),gsub1,vec=paste0(modality,'\\+'),vec1=colnames(X)))
x2 <- t(apply(X, 1, combn, 2, prod))
colnames(x2) <-paste(combn(modality, 2, paste, collapse="*"),combn(modality2, 2, paste, collapse="*"),sep='+')
#colnames(x2) <- paste("Inter.", combn(modality2, 2, paste, collapse="."), sep="")
X=cbind(X,x2)
out='out/'
AUC<-P0<-NULL
p=dim(X)[2]
for(i in 1:p)
{
rocobj <- pROC::auc(y,X[,i],smooth=F,direction ='auto')
temp=wilcox.test(X[y==1,i],X[y==0,i])$p.value
AUC<-c(AUC,max(rocobj,1-rocobj))
P0=c(P0,(temp))
}
library(qvalue)
Q0=qvalue(P0)$qvalues
INDDD=intersect(which(AUC>0.65),53:1378)
Q0=Q0[INDDD]
X=X[,INDDD]
p=dim(X)[2]

out='out/'
AUC<-P0<-NULL
for(i in 1:p)
{
rocobj <- pROC::auc(y,X[,i],smooth=F,direction ='auto')
temp=wilcox.test(X[y==1,i],X[y==0,i])$p.value
AUC<-c(AUC,max(rocobj,1-rocobj))
P0=c(P0,(temp))
}
library(qvalue)
stat0=data.frame(list(COR=AUC,pval=P0,qval=Q0))
modality=unlist(lapply(strsplit(colnames(X),'\\+'),'[',1))
modality2=unlist(lapply(strsplit(colnames(X),'\\+'),'[',2))
rownames(stat0)=paste0(modality,'+',modality2)
PATH0='out/circ.pdf'
temp0=sort(-stat0$COR,ind=T)$ix
stat0=stat0[temp0,]
##################################################
circplot<-function(stat0,PATH0='out/circ.pdf')
{
	factorstemp=substr(as.character(stat0$COR+rnorm(length(stat0$COR),0,0.001)),1,7)
	factors2=factor(factorstemp,levels =factorstemp)
	gsub1<-function(i,vec,vec1){return(gsub(vec[i],'',vec1[i]))}
	modality=unlist(lapply(strsplit(rownames(stat0),'\\+'),'[',1))
	factors3=unlist(lapply(strsplit(rownames(stat0),'\\+'),'[',2))
	grDevices::cairo_pdf(PATH0,width=12.5,height=13)
	par(mar=c(10.2,11,10.2,11), xpd=T)
	par(bg = "white")
	rbPal <- colorRampPalette(c("green", "yellow", "red"))
	allh <- stat0$COR
	col_hc<- rbPal(25)[as.numeric(cut(allh,breaks = 25))]
	col_hf<- rbPal(25)[as.numeric(cut(-log(stat0$pval)/log(10),breaks = 25))]
	col_hm<- rbPal(25)[as.numeric(cut(-log(stat0$qval)/log(10),breaks = 25))]
	par(bg = "white")
	circos.par("track.height" = 0.2)
	circos.initialize(factors2,xlim=c(0,3))
	print('succeed!!!')
	circos.track(factors = factors2,ylim = c(0, 1), panel.fun = function(x, y) {
	  chr = get.cell.meta.data("sector.index")
	  xlim = get.cell.meta.data("xlim")
	  ylim = get.cell.meta.data("ylim")
	  circos.rect(0, xlim[1], xlim[2], 1,border = NA
	  )
	  circos.text(mean(xlim),mean(ylim)-0.3,substr(allh,1,5)[which(factorstemp==chr)], 
	  cex = 1.3, adj = c(0, degree(0)),facing = "clockwise", niceFacing = T)
	  circos.text(mean(xlim),mean(ylim)+0.7,factors3[which(factorstemp==chr)], cex = 1.3, 
	  adj = c(0, degree(0)),facing = "clockwise", niceFacing = T)
	}, bg.border = NA,bg.col=col_hc)
	print('succeed!!!')
	circos.track(factors = factors2,ylim = c(0, 1), panel.fun = function(x, y) {
	  chr = get.cell.meta.data("sector.index")
	  xlim = get.cell.meta.data("xlim")
	  ylim = get.cell.meta.data("ylim")
	  circos.rect(0, xlim[1], xlim[2], 1,border = NA)
	  circos.text(mean(xlim),mean(ylim)-0.3, labels=substr(-log(stat0$pval)/log(10),1,5)[which(factorstemp==chr)], chr, cex = 1.3, adj = c(0, degree(0)),facing = "clockwise", niceFacing = T)
	  #print(get.cell.meta.data("xlim"))
	}, bg.border = NA,bg.col=col_hf)
	circos.track(factors = factors2,ylim = c(0, 1), panel.fun = function(x, y) {
	  chr = get.cell.meta.data("sector.index")
	  xlim = get.cell.meta.data("xlim")
	  ylim = get.cell.meta.data("ylim")
	  circos.rect(0, xlim[1], xlim[2], 1,border = NA)
	  circos.text(mean(xlim),mean(ylim)-0.3, labels=substr(-log(stat0$qval)/log(10),1,5)[which(factorstemp==chr)], chr, cex = 1.3, adj = c(0, degree(0)),facing = "clockwise", niceFacing = T)
	  #print(get.cell.meta.data("xlim"))
	}, bg.border = NA,bg.col=col_hm)
	circos.info()
	col_fun = colorRamp2(c(0,0.4,0.8), c("green", "yellow", "red"))
	tt1=-log(stat0$pval)/log(10)
	tt2=-log(stat0$qval)/log(10)
	A0=c(round(min(stat0$COR),2), round((min(stat0$COR)+max(stat0$COR))/2,2),round(max(stat0$COR),2))
	#lgd_links = Legend(at = A0,labels=A0, col_fun=colorRamp2(A0, c("green", "yellow", "red")),grid_width=NULL,legend_width=1,title_position = "topleft", title = "Outer: correlation", direction = "horizontal")
	A0=c(round(min(tt1),1), round((min(tt1)+max(tt1))/2,1),round(max(tt1),1))
	#lgd_links2 = Legend(at = A0,labels=A0, col_fun=colorRamp2(A0, c("green", "yellow", "red")),grid_width=NULL,legend_width=1,title_position = "topleft", title = expression("Middle: -log"["10"]*"p"), direction = "horizontal")
	A0=c(round(min(tt2),1), round((min(tt2)+max(tt2))/2,1),round(max(tt2),1))
	#lgd_links3 = Legend(at = A0,labels=A0, col_fun=colorRamp2(A0, c("green", "yellow", "red")),grid_width=NULL,legend_width=1,title_position = "topleft", title = expression("Inner: -log"["10"]*"q"), direction = "horizontal")
	#lgd_links = Legend(at = c(0.01, 0.4,0.8), col_fun=col_fun,title_position = "topleft", title = "Outer: correlation", direction = "horizontal")
	#lgd_links2 = Legend(at = c(0.01, 0.4,0.8), col_fun=col_fun,title_position = "topleft", title = expression("Middle: -log"["10"]*"p"), direction = "horizontal")
	#lgd_links3 = Legend(at = c(0.01, 0.4,0.8),col_fun=col_fun,title_position = "topleft", title = expression("Inner: -log"["10"]*"q"), direction = "horizontal")

	#lgd_list_vertical<-packLegend(lgd_links,lgd_links2,lgd_links3)
	#lgd_list_vertical
	#pushViewport(viewport(x = unit(20, "mm"), y = unit(30, "mm"),   width = grobWidth(lgd_list_vertical),  height = grobHeight(lgd_list_vertical), just = c("left", "bottom")))
	#grid.draw(lgd_list_vertical)
	#upViewport()
	circos.clear()
	dev.off()
}
circplot(stat0)