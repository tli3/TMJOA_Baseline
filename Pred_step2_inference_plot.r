#module load r/3.5.2;R
data0=read.csv('OAI_20190621.csv')
y=data0[,1]
STAT<-function(pred,y)
{
acc=sum((pred>0.5)==y)/length(y) 
prec1=sum((pred>0.5)&(y==1))/(sum(pred>0.5)+.00001) 
prec0=sum((pred<=0.5)&(y==0))/(sum(pred<=0.5)+.00001)  
recall1=sum((pred>0.5)&(y==1))/(sum(y==1)+.00001) 
recall0=sum((pred<0.5)&(y==0))/(sum(y==0)+.00001) 
auc0=pROC::roc(y,pred,smooth=F) 
f1score=(1/(1/prec1+1/recall1)+1/(1/prec0+1/recall0))
stat0=c(acc,prec1,prec0,recall1,recall0,as.numeric(auc0$auc),f1score) 
names(stat0)=c("ACC","PREC1","PREC0","RECALL1","RECALL0","AUC","f1score")
return(stat0)
}
A=read.csv('xgb/eta0.01W2C0.7S0.5/out/PredT.csv')
A=read.csv('LGB/eta0.01W1C0.7S0.5/out/PredT.csv')
rbind(apply(apply(A,2,STAT,y=y),1,mean),apply(apply(A,2,STAT,y=y),1,sd))
A=read.csv('xgb/eta0.01W1C0.5S0.5/out/PredT.csv')
B=read.csv('LGB/eta0.01W2C0.7S0.5/out/PredT.csv')
rbind(apply(apply((A+B)/2,2,STAT,y=y),1,mean),apply(apply((A+B)/2,2,STAT,y=y),1,sd))
#A=read.csv('xgb/final_0.95/out/PredT.csv')
#A1=read.csv('xgb/final_0.9/out/PredT.csv')
#A2=read.csv('xgb/final_0.8/out/PredT.csv')
#B=read.csv('LGB/final_0.95/out/PredT.csv')
#B1=read.csv('LGB/final_0.9/out/PredT.csv')
#B2=read.csv('LGB/final_0.8/out/PredT.csv')
#rbind(apply(apply((A+A1+B1+B+A2+B2)/6,2,STAT,y=y),1,mean),apply(apply((A+A1+B1+B+A2+B2)/6,2,STAT,y=y),1,sd))
#rbind(apply(apply((A2+B2)/2,2,STAT,y=y),1,mean),apply(apply((A2+B2)/2,2,STAT,y=y),1,sd))
#rbind(apply(apply((A+A1+B1+B)/4,2,STAT,y=y),1,mean),apply(apply((A+A1+B1+B)/4,2,STAT,y=y),1,sd))
#rbind(apply(apply((A+A1+A2+B)/4,2,STAT,y=y),1,mean),apply(apply((A+A1+A2+B)/4,2,STAT,y=y),1,sd))
########################################################################################
#Plot of Top features and their Importance
########################################################################################
#module load r/3.5.2;R
library(ggplot2)
A=read.csv('AllT.csv',check.names = FALSE)
L=dim(A)
Y=A[,L[2]]
X.fea=A[,-L[2]]
out='xgb/eta0.01W1C0.5S0.5/out';Method='XGBoost'
#out='LGB/eta0.01W2C0.7S0.5/out';Method='LightGBM'
SCORE=read.csv(paste0(out,'/importanceT.csv'),check.names = FALSE)
rownames(SCORE)=SCORE[,1];SCORE=SCORE[,-1];SCORE=t(SCORE)
tempp=sort(apply(SCORE,2,mean),decreasing=T,ind=T) 
mscore=tempp$x
iscore=tempp$ix
SCORE1=SCORE[,iscore]
cmscore=cumsum(mscore) 
cmIND=which(cmscore>0.8)[1] 
SCORE2=SCORE1[,1:cmIND]
feature0=matrix(NA,dim(A)[1],dim(SCORE2)[2])
colnames(feature0)=colnames(SCORE2)
ll=dim(feature0)
grDevices::cairo_pdf(paste0(out,'/1.pdf'))
par(mfcol=c(2,1),mai = c(0.05, 0.55, 0.1, 0.05))
options(scipen=999)
AUC<-P0<-NULL
for(i in 1:cmIND)
{
feature0[,i]=X.fea[names(mscore[i])][[1]]
rocobj <- pROC::roc(as.numeric(Y),feature0[,i],smooth=F)
temp=t.test(feature0[Y==1,i],feature0[Y==0,i])$p.value
AUC=c(AUC,as.numeric(rocobj$auc))
P0=c(P0,temp)
} 
feature20=(feature0-rep(1,ll[1])%*%t(apply(feature0,2,mean)))/(rep(1,ll[1])%*%t(apply(feature0,2,sd)))
feature30=as.matrix(feature20)[T]


library(qvalue)
Q0=qvalue(P0,pi0=1)$qvalues
data11=cbind(rep(colnames(feature0)[1:cmIND],3),c(AUC,-log10(P0),-log10(Q0)),
c(rep('AUC',length(AUC)),rep('Log10Pval',length(AUC)),rep('Log10Qval',length(AUC))),rep(mscore[1:cmIND],3))
colnames(data11)=c('Feature','Y','Measure','mscore0');data11=data.frame(data11)
data11[,2]=as.numeric(as.character(data11[,2]))
data11[,4]=as.numeric(as.character(data11[,4]))
class0=rep(Y,ll[2])
class1=ifelse(class0==1,"OA","Normal")
name0=t(matrix(rep(colnames(feature0),ll[1]),ll[2]))[T]
data00=cbind(name0,class1,feature30,as.matrix(SCORE2)[T])
colnames(data00)=c('Feature','Group','Score','Score0')
data00=data.frame(data00)
data00$Score=as.numeric(as.character(data00$Score))
data00$Score0=as.numeric(as.character(data00$Score0))
data00$Feature=unlist(lapply(strsplit(as.character(data00$Feature),'\\+'),'[',2))
data00$Feature=as.factor(data00$Feature)
temp0=levels(data00$Feature)
temp0[as.numeric(unique(data00$Feature))]=paste0(temp0[as.numeric(unique(data00$Feature))],' (AUC: ',round(AUC,3),')')
levels(data00$Feature)=temp0
data00$Group=as.factor(data00$Group)


ggplot(aes(y = Score, x = reorder(Feature,-Score0,mean), fill = Group), data = data00) +
geom_boxplot()+ggtitle(paste0("Boxplots of top features for OA vs Normal"))+
theme(axis.text.x = element_text(face="bold",angle = 90, hjust = 1),
plot.title = element_text(lineheight=.8, face="bold",hjust=0.5))+xlab('')+ylab('Normalized Measures')
dev.off()

###############################################################
grDevices::cairo_pdf(paste0(out,"/Boxplot_CV.pdf")) 
SCORE2=SCORE1[,1:cmIND] 
nn=t(matrix(rep(colnames(SCORE2),dim(SCORE2)[1]),dim(SCORE2)[2]))[T] 
temp000=as.matrix(SCORE2)[T] 
data00=cbind(nn,as.numeric(temp000));data00=data.frame(data00); 
data00[,1]=as.character(data00[,1])
data00[[2]]=as.numeric(as.character(data00[[2]]));colnames(data00)=c("TopFeatures","Model_Contributions_CV")
data00$TopFeatures=unlist(lapply(strsplit(as.character(data00$TopFeatures),'\\+'),'[',2))
means <- aggregate(Model_Contributions_CV ~  TopFeatures, data00, mean)
print(eval(substitute(ggplot(data00, aes(x=reorder(var1,var2,mean), y=var2)) +geom_boxplot(color="tan4",fill="darkslateblue",alpha=0.2, 
        size=0.5,notch=TRUE,notchwidth = 0.3,outlier.colour="red",outlier.fill="red",outlier.size=1)+ 
		ggtitle(paste0("Contribution of top ",cmIND," features (>80%) in ",Method,""))+
		stat_summary(fun.y=mean, geom="point", shape=20, size=1, color="red", fill="red") +
		theme(axis.text.x = element_text(face="bold",angle = 90, hjust = 1),plot.title = element_text(lineheight=.8, face="bold",hjust=0.5))+
		xlab('')+coord_flip(),
		list(var1=as.name(colnames(data00)[1]),var2=as.name(colnames(data00)[2]))))) 
dev.off() 
###############################################################
#AUCplot
###############################################################
ss0=NULL
temp0=read.csv('xgb/eta0.01W1C0.5S0.5/out/PredT.csv')
for(i in 1:10){ss0=c(ss0,as.numeric(pROC::roc(Y,temp0[,i])$auc))}
pred_XGB=apply(temp0,1,mean)
ss1=NULL
temp1=read.csv('LGB/eta0.01W2C0.7S0.5/out/PredT.csv')
for(i in 1:10){ss1=c(ss1,as.numeric(pROC::roc(Y,temp1[,i])$auc))}
pred_LGB=apply(temp1,1,mean)
ss2=NULL
temp2=(temp0+temp1)/2
for(i in 1:10){ss2=c(ss2,as.numeric(pROC::roc(Y,temp2[,i])$auc))}
pred_ENS=apply(temp2,1,mean)
grDevices::cairo_pdf(paste0('./','rocT.pdf'))
par(mar=c(10.1, 4.1, 10.1, 9.1), xpd=TRUE)
type0=c(1,1,1,1,1,1,2,2,2,2,2)
type0=c(rep(1,6),rep(3,5))
smooth0=F
wid0=2.1
col0=c('black','gray47','blue','darkorange','red','darkgreen','purple','mediumvioletred',
'darkred','darkgoldenrod4','slateblue4')
pROC::plot.roc(as.numeric(Y)~pred_XGB,smooth=smooth0,lty=type0[1],lwd=wid0,col=col0[1],cex.lab=1.3,cex.axis=1.3)
pROC::plot.roc(as.numeric(Y)~pred_LGB,smooth=smooth0,add=T,lty=type0[2],lwd=wid0,col=col0[2],cex.lab=1.3,
cex.axis=1.3)
pROC::plot.roc(as.numeric(Y)~pred_ENS,smooth=smooth0,add=T,lty=type0[3],lwd=wid0,col=col0[3],cex.lab=1.3,cex.axis=1.3)
for(i in 1:min(cmIND,8))
{
pROC::plot.roc(as.numeric(Y)~X.fea[names(mscore[i])][[1]],smooth=smooth0,lty=type0[i+3],
lwd=wid0,col=col0[i+3],cex.lab=1.3,cex.axis=1.3,add=T)
}
legend('bottomright', legend=c('LightGBM','XGB','LightGBM+XGB',
#legend('bottomright', legend=c('LightGBM+XGB',
unlist(lapply(strsplit(names(mscore)[1:min(cmIND,8)],'\\+'),'[[',2))),
       col=col0, lty=type0, cex=0.75,lwd=rep(2,min(length(cmIND),8)+3),
       text.font=1,box.lwd=0,box.col='white',bg='lightgray')
dev.off()


