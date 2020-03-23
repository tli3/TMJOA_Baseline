#module load r/3.5.2; R
library(Matrix)
library(ggplot2) # Data visualization
library(data.table)
library(xgboost)
library(caret)
source('manhattan.r')
A=read.csv('OAI_20190621.csv',check.names = FALSE)
y=A[,1]
X=A[,-1]
modality=unlist(lapply(strsplit(colnames(X),'\\+'),'[',1))
gsub1<-function(i,vec,vec1){return(gsub(vec[i],'',vec1[i]))}
modality2=unlist(lapply(1:length(modality),gsub1,vec=paste0(modality,'\\+'),vec1=colnames(X)))
x2 <- t(apply(X, 1, combn, 2, prod))
colnames(x2) <-paste(combn(modality, 2, paste, collapse="*"),combn(modality2, 2, paste, collapse="*"),sep='+')
X=cbind(X,x2)
p=dim(X)[2]
AUC<-P0<-NULL
for(i in 1:p)
{
rocobj <- pROC::roc(y,X[,i],smooth=F)
temp=wilcox.test(X[y==1,i],X[y==0,i])$p.value #t.tset
AUC<-c(AUC,max(rocobj$auc,1-rocobj$auc))
P0=c(P0,(temp))
}
library(qvalue)
Q0=qvalue(P0)$qvalues
X=X[,-(1:52)]
p=dim(X)[2]
XX=X
modality=unlist(lapply(strsplit(colnames(X),'\\+'),'[',1))
modality1=unlist(lapply(strsplit(modality,'\\*'),paste,collapse='\\*'))
gsub1<-function(i,vec,vec1){return(gsub(vec[i],'',vec1[i]))}
modality2=unlist(lapply(1:length(modality),gsub1,vec=paste0(modality1,'\\+'),vec1=colnames(X)))
modality31=unlist(lapply(strsplit(modality,'\\*'),'[',1))
modality32=unlist(lapply(strsplit(modality,'\\*'),'[',2))
INDDD=which(modality31!=modality32)
X=cbind(X,XX[,INDDD])
modality=paste0(c(modality31,modality32[INDDD]),'*')
#X=cbind(X[,c(1:1667,2208:2227,1668:2207,2228:2307)])
#modality=modality[c(1:1667,2208:2227,1668:2207,2228:2307)]
umod=paste0(c("Clinic","Serum","Saliva","Demo","Trabe"),'*')
INDD=sort(as.numeric(factor(modality,levels=umod)),ind=T)$ix
X=X[,INDD];modality=modality[INDD]
p=dim(X)[2]
modality1=unlist(lapply(strsplit(modality,'\\*'),paste,collapse='\\*'))
gsub1<-function(i,vec,vec1){return(gsub(vec[i],'',vec1[i]))}
modality2=unlist(lapply(strsplit(colnames(X),'\\+'),'[',2))
out='out/'
##############################################################################
pdf(paste0(out,'/mhplot.pdf'))
#grDevices::cairo_pdf(paste0(out,'/mhplot.pdf'),width=11,height=13)
#	par(mar=c(10.2,11,10.2,11), xpd=T)
#	par(bg = "white")	
#levels0=c("Clinic*else","Trabe*else","Serum*else","Saliva*else","Demo*else")
levels0=umod
cols<-c('green','forestgreen','sienna',
		'royalblue2','darkturquoise','darkviolet','orange','midnightblue',"red")
AUC<-P0<-NULL
for(i in 1:p)
{
rocobj <- pROC::roc(y,X[,i],smooth=F)
temp=wilcox.test(X[y==1,i],X[y==0,i])$p.value #t.tset
AUC<-c(AUC,max(rocobj$auc,1-rocobj$auc))
P0=c(P0,(temp))
}
library(qvalue)
Q0=qvalue(P0)$qvalues
bonf=P0
for(temp in umod){
print(temp)
bonf[modality==temp]=bonf[modality==temp]*sum(modality==temp)}
#Q0=bonf
snp0=rep(NA,length(AUC))
for(i in 1:length(AUC)){tempp=as.numeric(factor(modality,levels=levels0));snp0[i]=sum(tempp[1:i]==tempp[i])-1;snp0=snp0/1}
snp<-data.frame(cbind(as.numeric(factor(modality,levels=levels0)),
AUC,snp0,c(1:length(AUC))))
colnames(snp)<-c("CHR","P","BP","SNP")
snp$SNP=modality2
snp[,2]=10^(-snp[,2])
par(mfrow=c(3,1))
manhattan(snp,logp=T,col = cols,chrlabs=levels(factor(modality,levels=levels0)), las = 2,chr="CHR",ylab="AUC",	main="",
xlab="",suggestiveline = 0.6, genomewideline = 0.65,annotatePval=0.7,cex.axis = 1.3,ylim=c(0.45,0.8),cex.main=2,cex.lab=1.3)
snp0=rep(NA,length(AUC))
for(i in 1:length(AUC)){tempp=as.numeric(factor(modality,levels=levels0));snp0[i]=sum(tempp[1:i]==tempp[i])-1;snp0=snp0/1}
snp<-data.frame(cbind(as.numeric(factor(modality,levels=levels0)),
P0,snp0,c(1:length(AUC))))
colnames(snp)<-c("CHR","P","BP","SNP")
snp$SNP=modality2
manhattan(snp,logp=T,col = cols,chrlabs=levels(factor(modality,
levels=levels0)), las = 2,
		  chr="CHR",ylab="-Log10[Pval] (Wilcoxon)",
		  main=paste(""),
		  xlab="",suggestiveline = 1.301, genomewideline = 2,annotatePval=0.05,cex.axis = 1.3,cex.lab=1.3)
snp0=rep(NA,length(AUC))
for(i in 1:length(AUC)){tempp=as.numeric(factor(modality,levels=levels0));snp0[i]=sum(tempp[1:i]==tempp[i])-1;snp0=snp0/1}
snp<-data.frame(cbind(as.numeric(factor(modality,levels=levels0)),
Q0,snp0,c(1:length(AUC))))
colnames(snp)<-c("CHR","P","BP","SNP")
snp$SNP=modality2
manhattan(snp,logp=T,col = cols,chrlabs=levels(factor(modality,
levels=levels0)), las = 2,
		  chr="CHR",ylab="-Log10[Qval]",
		  main=paste(""),
		  xlab="",suggestiveline = 1, 
		  genomewideline = 1.301,cex.axis = 1.3,cex.lab=1.3,annotatePval=0.1)
dev.off()
