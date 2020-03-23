#module load r/3.5.2; R
A=read.csv('OAI_20190621.csv',check.names = FALSE)
y=A[,1]
X=A[,-1]
p=dim(X)[2]
modality=unlist(lapply(strsplit(colnames(X),'\\+'),'[',1))
gsub1<-function(i,vec,vec1){return(gsub(vec[i],'',vec1[i]))}
modality2=unlist(lapply(1:length(modality),gsub1,vec=paste0(modality,'\\+'),vec1=colnames(X)))
umod=unique(modality)
##################################################
Mod=umod;Mod=setdiff(Mod,'Demo');Mod1=NULL
for(i in 1:length(Mod))
{
Mod1=c(Mod1,combn(Mod,i,simplify=F))
}
Mod1=lapply(Mod1,c,'Demo')
Y=y
l=length(y)
X.fea=X
pred=rep(NA,l)
x2 <- t(apply(X.fea, 1, combn, 2, prod))
colnames(x2) <-paste(combn(modality, 2, paste, collapse="*"),combn(modality2, 2, paste, collapse="*"),sep='+')
#colnames(x2) <- paste("Inter.", combn(modality2, 2, paste, collapse="."), sep="")
X.fea=cbind(X.fea,x2)
sss=cbind(X.fea,Y)
write.csv(sss,file='AllT.csv',quote=F,row.names=F)
##########################################################
A0T=A=read.csv('AllT.csv',check.names = FALSE)
L=dim(A)
y=A[,L[2]]
A=A[,-L[2]]
AUC=matrix(NA,dim(A)[2],10)
colnames(AUC)=as.character(2020:2029)
rownames(AUC)=colnames(A)
for(i in 1:5)
{
	for(seed1 in 2020:2029)
	{
		set.seed(seed1)
		foldsCV <- createFolds(y, k=5, list=TRUE, returnTrain=FALSE)
		Y1=y[-foldsCV[[i]]]
		X.fea0=A[-foldsCV[[i]],]
		p=dim(X.fea0)[2]
		X.fea00=X.fea0
		for(j in 1:p){if(j%%2000==0)print(p-j);AUC[j,seed1-2019]=pROC::roc(Y1,X.fea00[,j],smooth=F,quiet=T)$auc}
	}
	write.csv(AUC,paste0('AUC_i',i,'.csv'),quote=F)
}
AUC1=read.csv('AUC_i1.csv')
AUC2=read.csv('AUC_i2.csv')
AUC3=read.csv('AUC_i3.csv')
AUC4=read.csv('AUC_i4.csv')
AUC5=read.csv('AUC_i5.csv')
AUCT=cbind(AUC1[,-1],AUC2[,-1],AUC3[,-1],AUC4[,-1],AUC5[,-1])
rownames(AUCT)=AUC1[,1]
colnames(AUCT)=c(paste0(2020:2029,'_1'),paste0(2020:2029,'_2'),paste0(2020:2029,'_3'),
paste0(2020:2029,'_4'),paste0(2020:2029,'_5'))
write.csv(AUCT,'AUCT.csv',quote=F)
#######################################################
