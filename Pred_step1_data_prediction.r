##################################################################
#module load r/3.5.2; R
#Random Forest
library(Matrix)
library(qqman)
library(ggplot2) # Data visualization
library(data.table)
library(xgboost)
library(caret)
options(scipen=999)
library(data.table)
library(randomForest)
library(caret)
A=read.csv('AllT.csv',check.names = FALSE)
L=dim(A)
y=Y=A[,L[2]]
X=A[,-L[2]]
p=dim(X)[2]	
for(kk in 1:1)
{
	out='RF/'
	system(paste0('mkdir -p ',out,'/out'))
	X.fea=X;	STATT=matrix(NA,10,7);colnames(STATT)=c("ACC","PREC1","PREC0","RECALL1","RECALL0","F1SCORE","AUC") 
	AUCT=read.csv('AUCT.csv')
	SCORE=matrix(0,50,p)
	colnames(SCORE)=colnames(X.fea)
	PredT=matrix(NA,length(y),10)
	colnames(PredT)=2020:2029
	for(seed1 in 2020:2029)
	{
		print(seed1)
		set.seed(seed1)
		foldsCV <- createFolds(y, k=5, list=TRUE, returnTrain=FALSE)
		pred=y;pred[T]=NA
		for(i in 1:5)
		{
			set.seed(0)
			Y1=y[-foldsCV[[i]]]
			X.fea0=X.fea[-foldsCV[[i]],]
			AUC=AUCT[paste0('X',seed1,'_',i)][,1]
			inddd=which(AUC>0.70)
			X.fea0=X.fea0[,inddd]
			rf0 <- randomForest(x=X.fea0,y=as.factor(Y1), importance=TRUE,proximity=TRUE)
			final=as.matrix(rf0$importance)
			aaa=final[,3];SCORE[i+(seed1-2020)*5,names(aaa)]=aaa
			pred[foldsCV[[i]]] <- as.numeric(as.character(predict(rf0,as.matrix(X.fea[foldsCV[[i]],inddd]))))
		}
		tos=cbind(y,pred) 
		colnames(tos)=c("Y","Pred") 
		acc=sum((pred>0.5)==Y)/length(Y) 
		prec1=sum((pred>0.5)&(Y==1))/(sum(pred>0.5)+.00001) 
		prec0=sum((pred<=0.5)&(Y==0))/(sum(pred<=0.5)+.00001) 
		recall1=sum((pred>0.5)&(Y==1))/(sum(Y==1)+.00001) 
		recall0=sum((pred<0.5)&(Y==0))/(sum(Y==0)+.00001) 
		auc0=pROC::roc(Y,pred,smooth=F) 
		f1score=(1/(1/prec1+1/recall1)+1/(1/prec0+1/recall0))
		statt0=c(acc,prec1,prec0,recall1,recall0,f1score,as.numeric(auc0$auc)) 
		STATT[seed1-2019,]=t(round(statt0,4))
		PredT[,seed1-2019]=pred
		print(STATT[seed1-2019,])
	}
	write.csv(PredT,file=paste0(out,'out/PredT.csv'),row.names=F,quote=F)
	write.csv(t(SCORE),file=paste0(out,'out/importanceT.csv'),quote=F)
	write.csv(STATT,file=paste0(out,'out/STATT.csv'),row.names=F,quote=F)
	tempp=sort(apply(SCORE,2,mean),decreasing=T,ind=T) 
	mscore=tempp$x 
	iscore=tempp$ix 
	SCORE1=SCORE[,iscore] 
	write.table(mscore,paste0(out,"/importance.txt"),quote=F)
}
##################################################################
#module load r/3.5.2; R
library(Matrix)
library(qqman)
library(ggplot2) # Data visualization
library(data.table)
library(xgboost)
library(caret)
options(scipen=999)
library(data.table)
library(xgboost)
library(caret)
A=read.csv('AllT.csv',check.names = FALSE)
L=dim(A)
y=Y=A[,L[2]]
X=A[,-L[2]]
p=dim(X)[2]	
PARA=matrix(c(0.001,0.001,0.01,0.001,2,1,2,2,0.7,0.7,0.7,0.5,
0.5,0.5,0.5,0.5),4)
PARA=matrix(c(0.01,0.01,0.01,1,2,1,0.7,0.5,0.5,0.5,0.5,0.5),3)
for(kk in 1:4)
{
	eta=PARA[kk,1]
	W=PARA[kk,2]
	C=PARA[kk,3]
	S=PARA[kk,4]
	out=paste0('xgb/eta',eta,'W',W,'C',C,'S',S,'/')
	system(paste0('mkdir -p ',out,'/out'))
	X.fea=X;	STATT=matrix(NA,10,7);colnames(STATT)=c("ACC","PREC1","PREC0","RECALL1","RECALL0","F1SCORE","AUC") 
	AUCT=read.csv('AUCT.csv')
	SCORE=matrix(0,50,p)
	colnames(SCORE)=colnames(X.fea)
	PredT=matrix(NA,length(y),10)
	colnames(PredT)=2020:2029
	for(seed1 in 2020:2029)
	{
		print(seed1)
		set.seed(seed1)
		foldsCV <- createFolds(y, k=5, list=TRUE, returnTrain=FALSE)
		pred=y;pred[T]=NA
		for(i in 1:5)
		{
			set.seed(0)
			Y1=y[-foldsCV[[i]]]
			X.fea0=X.fea[-foldsCV[[i]],]
			AUC=AUCT[paste0('X',seed1,'_',i)][,1]
			inddd=which(AUC>0.70)
			X.fea0=X.fea0[,inddd]
			dtrain=xgb.DMatrix(data=as.matrix(X.fea0),label=Y1, missing=NA)
			dtest=xgb.DMatrix(data=as.matrix(X.fea[foldsCV[[i]],inddd]),label=Y[foldsCV[[i]]], missing=NA)
			foldsCV0 <- createFolds(Y1, k=5, list=TRUE, returnTrain=FALSE)
			param <- list(objective = "binary:logistic"
					  , subsample = S
					  , max_depth = 1
					  , colsample_bytree = C
					  , eta = eta
					  , eval_metric = 'auc'#f1score_eval# map: original
					  , min_child_weight = W,scale_pos_weight=sum(Y1==0)/sum(Y1==1)) 
			xgb_cv <- xgb.cv(data=dtrain,params=param,nrounds=10000,prediction=TRUE,maximize=TRUE,
			folds=foldsCV0,verbose=F)
			temp=as.matrix(xgb_cv$evaluation_log[,4])
			nrounds=which.max(temp)
			xgb <- xgb.train(params = param, data = dtrain, nrounds = nrounds, verbose = 0)
			final=as.matrix(xgb.importance(model=xgb))
			aaa=as.numeric(final[,2]);names(aaa)=final[,1]
			SCORE[i+(seed1-2020)*5,names(aaa)]=aaa
			pred[foldsCV[[i]]] <- predict(xgb,dtest)
		}
		tos=cbind(y,pred) 
		colnames(tos)=c("Y","Pred") 
		acc=sum((pred>0.5)==Y)/length(Y) 
		prec1=sum((pred>0.5)&(Y==1))/(sum(pred>0.5)+.00001) 
		prec0=sum((pred<=0.5)&(Y==0))/(sum(pred<=0.5)+.00001) 
		recall1=sum((pred>0.5)&(Y==1))/(sum(Y==1)+.00001) 
		recall0=sum((pred<0.5)&(Y==0))/(sum(Y==0)+.00001) 
		auc0=pROC::roc(Y,pred,smooth=F) 
		f1score=(1/(1/prec1+1/recall1)+1/(1/prec0+1/recall0))
		statt0=c(acc,prec1,prec0,recall1,recall0,f1score,as.numeric(auc0$auc)) 
		STATT[seed1-2019,]=t(round(statt0,4))
		PredT[,seed1-2019]=pred
		print(STATT[seed1-2019,])
	}
	write.csv(PredT,file=paste0(out,'out/PredT.csv'),row.names=F,quote=F)
	write.csv(t(SCORE),file=paste0(out,'out/importanceT.csv'),quote=F)
	write.csv(STATT,file=paste0(out,'out/STATT.csv'),row.names=F,quote=F)
	tempp=sort(apply(SCORE,2,mean),decreasing=T,ind=T) 
	mscore=tempp$x 
	iscore=tempp$ix 
	SCORE1=SCORE[,iscore] 
	write.table(mscore,paste0(out,"/importance.txt"),quote=F)
}

##################################################################
#module load r/3.5.2; R
library(Matrix)
library(qqman)
library(MLmetrics)
library(ggplot2) # Data visualization
library(data.table)
library(caret)
library(lightgbm)
options(scipen=999)
library(data.table)
A=read.csv('AllT.csv',check.names = FALSE)
L=dim(A)
y=Y=A[,L[2]]
X=A[,-L[2]]
p=dim(X)[2]	
PARA=matrix(c(0.01,0.01,0.001,0.01,0.01,0.01,0.01,
1,1,1,1,2,2,2,0.5,0.7,0.7,0.7,0.7,0.5,0.7,
0.5,0.5,0.5,0.7,0.5,0.5,0.7),7)
PARA=matrix(c(0.01,1,0.5,0.7),1)
for(kk in 1:1)
{
	eta=PARA[kk,1]
	W=PARA[kk,2]
	C=PARA[kk,3]
	S=PARA[kk,4]
	out=paste0('LGB/eta',eta,'W',W,'C',C,'S',S,'/')
	system(paste0('mkdir -p ',out,'/out'))
	X.fea=X;	STATT=matrix(NA,10,7);colnames(STATT)=c("ACC","PREC1","PREC0","RECALL1","RECALL0","F1SCORE","AUC") 
	AUCT=read.csv('AUCT.csv')
	SCORE=matrix(0,50,p)
	colnames(SCORE)=colnames(X.fea)
	PredT=matrix(NA,length(y),10)
	colnames(PredT)=2020:2029
	for(seed1 in 2020:2029)
	{
		print(seed1)
		set.seed(seed1)
		foldsCV <- createFolds(y, k=5, list=TRUE, returnTrain=FALSE)
		pred=y;pred[T]=NA
		for(i in 1:5)
		{
			set.seed(0)
			Y1=y[-foldsCV[[i]]]
			X.fea0=X.fea[-foldsCV[[i]],]
			AUC=AUCT[paste0('X',seed1,'_',i)][,1]
			inddd=which(AUC>0.71)
			X.fea0=X.fea0[,inddd]
			dtrain=lgb.Dataset(as.matrix(X.fea0), label=Y1,missing=NA)
			foldsCV0 <- createFolds(Y1, k=5, list=TRUE, returnTrain=FALSE)
			param = list(objective = "binary",metric = "auc",
                min_sum_hessian_in_leaf = W,feature_fraction = C,learning_rate = eta, max_depth=1,
                bagging_fraction = S,is_unbalance = T)
			lgb_cv = lgb.cv(params =param, data = dtrain,num_threads = 2 , 
			nrounds = 500,eval = "auc",folds=foldsCV0,stratified = TRUE,verbose=-1)
			#nrounds = 500,eval = "auc",folds=foldsCV0,stratified = TRUE,verbose=-1)
			nrounds=lgb_cv$best_iter
			lgb <-lgb.train(params = param, data = dtrain,nrounds = nrounds,verbose=-1)
			final=as.matrix(lgb.importance(model=lgb))
			aaa=as.numeric(final[,2]);names(aaa)=final[,1]
			SCORE[i+(seed1-2020)*5,names(aaa)]=aaa
			pred[foldsCV[[i]]] <- predict(lgb,as.matrix(X.fea[foldsCV[[i]],inddd]))
		}
		tos=cbind(y,pred) 
		colnames(tos)=c("Y","Pred") 
		acc=sum((pred>0.5)==Y)/length(Y) 
		prec1=sum((pred>0.5)&(Y==1))/(sum(pred>0.5)+.00001) 
		prec0=sum((pred<=0.5)&(Y==0))/(sum(pred<=0.5)+.00001) 
		recall1=sum((pred>0.5)&(Y==1))/(sum(Y==1)+.00001) 
		recall0=sum((pred<0.5)&(Y==0))/(sum(Y==0)+.00001) 
		auc0=pROC::roc(Y,pred,smooth=F) 
		f1score=(1/(1/prec1+1/recall1)+1/(1/prec0+1/recall0))
		statt0=c(acc,prec1,prec0,recall1,recall0,f1score,as.numeric(auc0$auc)) 
		STATT[seed1-2019,]=t(round(statt0,4))
		PredT[,seed1-2019]=pred
		print(STATT[seed1-2019,])
	}
	write.csv(PredT,file=paste0(out,'out/PredT.csv'),row.names=F,quote=F)
	write.csv(t(SCORE),file=paste0(out,'out/importanceT.csv'),quote=F)
	write.csv(STATT,file=paste0(out,'out/STATT.csv'),row.names=F,quote=F)
	tempp=sort(apply(SCORE,2,mean),decreasing=T,ind=T) 
	mscore=tempp$x 
	iscore=tempp$ix 
	SCORE1=SCORE[,iscore] 
	write.table(mscore,paste0(out,"/importance.txt"),quote=F)
}
##################################################################
# module load r/3.5.2; R
library(Matrix)
library(qqman)
library(ggplot2) # Data visualization
library(data.table)
library(xgboost)
library(caret)
options(scipen=999)
library(data.table)
library(xgboost)
library(caret)
library(glmnet)
A=read.csv('AllT.csv',check.names = FALSE)
L=dim(A)
y=Y=A[,L[2]]
X=A[,-L[2]]
p=dim(X)[2]	
out=paste0('Glmnet/','/')
system(paste0('mkdir -p ',out,'/out'))
X.fea=X;	STATT=matrix(NA,10,7);colnames(STATT)=c("ACC","PREC1","PREC0","RECALL1","RECALL0","F1SCORE","AUC") 
AUCT=read.csv('AUCT.csv')
PredT=matrix(NA,length(y),10)
colnames(PredT)=2020:2029
for(seed1 in 2020:2029)
{
	print(seed1)
	set.seed(seed1)
	foldsCV <- createFolds(y, k=5, list=TRUE, returnTrain=FALSE)
	pred=y;pred[T]=NA
	for(i in 1:5)
	{
		set.seed(0)
		Y1=y[-foldsCV[[i]]]
		X.fea0=X.fea[-foldsCV[[i]],]
		AUC=AUCT[paste0('X',seed1,'_',i)][,1]
		inddd=which(AUC>0.70)
		X.fea0=X.fea0[,inddd]
		cv.ridge <- cv.glmnet(as.matrix(X.fea0), as.double(Y1), family='binomial', alpha=0,  standardize=TRUE,type.measure='auc',nfolds=5)
		model0<-glmnet(as.matrix(X.fea0), as.double(Y1), family='binomial', alpha=0, standardize=TRUE,lambda=cv.ridge$lambda.min)
		temp11<-coef(model0); temp11<-exp(as.matrix(X.fea[foldsCV[[i]],inddd])%*%temp11[-1]+temp11[1])
		pred[foldsCV[[i]]] <- temp11/(1+temp11)
	}
	tos=cbind(y,pred) 
	colnames(tos)=c("Y","Pred") 
	acc=sum((pred>0.5)==Y)/length(Y) 
	prec1=sum((pred>0.5)&(Y==1))/(sum(pred>0.5)+.00001) 
	prec0=sum((pred<=0.5)&(Y==0))/(sum(pred<=0.5)+.00001) 
	recall1=sum((pred>0.5)&(Y==1))/(sum(Y==1)+.00001) 
	recall0=sum((pred<0.5)&(Y==0))/(sum(Y==0)+.00001) 
	auc0=pROC::roc(Y,pred,smooth=F) 
	f1score=(1/(1/prec1+1/recall1)+1/(1/prec0+1/recall0))
	statt0=c(acc,prec1,prec0,recall1,recall0,f1score,as.numeric(auc0$auc)) 
	STATT[seed1-2019,]=t(round(statt0,4))
	PredT[,seed1-2019]=pred
	print(STATT[seed1-2019,])
}
write.csv(PredT,file=paste0(out,'out/PredT.csv'),row.names=F,quote=F)
write.csv(STATT,file=paste0(out,'out/STATT.csv'),row.names=F,quote=F)
##################################################################
#module load r/3.5.2; R
library(Matrix)
library(qqman)
library(ggplot2) # Data visualization
library(data.table)
library(xgboost)
library(caret)
options(scipen=999)
library(data.table)
library(xgboost)
library(caret)
library(glmnet)
A=read.csv('AllT.csv',check.names = FALSE)
L=dim(A)
y=Y=A[,L[2]]
X=A[,-L[2]]
p=dim(X)[2]	
out=paste0('Logistic/','/')
system(paste0('mkdir -p ',out,'/out'))
X.fea=X[,1:52];	STATT=matrix(NA,10,7);colnames(STATT)=c("ACC","PREC1","PREC0","RECALL1","RECALL0","F1SCORE","AUC") 
AUCT=read.csv('AUCT.csv')
AUCT=AUCT[1:52,]
PredT=matrix(NA,length(y),10)
colnames(PredT)=2020:2029
for(seed1 in 2020:2029)
{
	print(seed1)
	set.seed(seed1)
	foldsCV <- createFolds(y, k=5, list=TRUE, returnTrain=FALSE)
	pred=y;pred[T]=NA
	for(i in 1:5)
	{
		set.seed(0)
		Y1=y[-foldsCV[[i]]]
		X.fea0=X.fea[-foldsCV[[i]],]
		AUC=AUCT[paste0('X',seed1,'_',i)][,1]
		inddd=which(AUC>0.70)
		X.fea0=X.fea0[,inddd]
		model0=glm(Y1~.+1,as.data.frame(X.fea0),family = "binomial");
		temp11=exp(as.matrix(X.fea[foldsCV[[i]],inddd])%*%model0$coef[-1]+model0$coef[1])
		pred[foldsCV[[i]]] <- temp11/(1+temp11)
	}
	tos=cbind(y,pred) 
	colnames(tos)=c("Y","Pred") 
	acc=sum((pred>0.5)==Y)/length(Y) 
	prec1=sum((pred>0.5)&(Y==1))/(sum(pred>0.5)+.00001) 
	prec0=sum((pred<=0.5)&(Y==0))/(sum(pred<=0.5)+.00001) 
	recall1=sum((pred>0.5)&(Y==1))/(sum(Y==1)+.00001) 
	recall0=sum((pred<0.5)&(Y==0))/(sum(Y==0)+.00001) 
	auc0=pROC::roc(Y,pred,smooth=F) 
	f1score=(1/(1/prec1+1/recall1)+1/(1/prec0+1/recall0))
	statt0=c(acc,prec1,prec0,recall1,recall0,f1score,as.numeric(auc0$auc)) 
	STATT[seed1-2019,]=t(round(statt0,4))
	PredT[,seed1-2019]=pred
	print(STATT[seed1-2019,])
}
write.csv(PredT,file=paste0(out,'out/PredT.csv'),row.names=F,quote=F)
write.csv(STATT,file=paste0(out,'out/STATT.csv'),row.names=F,quote=F)
##################################################################
#XGBoost with Fixed feature: Backward deletion					 #
##################################################################
#module load r/3.5.2;R
library(xgboost)
library(caret)
A0T=read.csv('AllT.csv')
L=dim(A0T)
y=Y=A0T[,L[2]]
A0T=A0T[,-L[2]]
out='xgb/eta0.01W1C0.5S0.5/out'
SCORE=read.csv(paste0(out,'/importanceT.csv'),check.names = FALSE)
rownames(SCORE)=SCORE[,1];SCORE=SCORE[,-1];SCORE=t(SCORE)
tempp=sort(apply(SCORE,2,mean),decreasing=T,ind=T) 
mscore=tempp$x
iscore=tempp$ix
SCORE1=SCORE[,iscore]
cmscore=cumsum(mscore) 
#for(start in c(0.8,0.9,0.95))
for(start in c(0.8))
{
	cmIND=min(which(cmscore>start)[1],30)
	selected=colnames(SCORE1)[1:cmIND]
	AUCT=read.csv('AUCT.csv')
	l0=dim(AUCT)[1]
	rownames(AUCT)=AUCT[,1]
	AUCT=AUCT[,-1]
	colnames(A0T)=row.names(AUCT)
	##################################################################
	A=A0T
	inddd=which(colnames(A0T)%in%selected)
	ACC0=0
	while(1)
	{
		ll=length(inddd)
		STATT0=matrix(NA,ll,9)
		colnames(STATT0)=c("ACC","PREC1","PREC0","RECALL1","RECALL0","f1score","AUC",'Sensitivity','Specificity')
		for(kk in 1:ll)
		{		
			print(ll-kk)
			STATT=matrix(NA,10,9);
			colnames(STATT)=c("ACC","PREC1","PREC0","RECALL1","RECALL0","f1score","AUC",'Sensitivity','Specificity')
			SGene=NULL;sg0=1;sen0=NULL
			for(seed1 in 2020:2029)
			{
				set.seed(seed1)
				foldsCV <- createFolds(y, k=5, list=TRUE, returnTrain=FALSE)
				########################################################
				p=dim(A)[2]
				pred=y;pred[T]=NA
				for(i in 1:5)
				{
					set.seed(0)
					Y1=y[-foldsCV[[i]]]
					X.fea0=A[-foldsCV[[i]],]
					p=dim(X.fea0)[2]
					X.fea00=X.fea0
					AUC=AUCT[paste0('X',seed1,'_',i)][,1]
					inddd1= inddd[-kk]
					X.fea0=X.fea00[,inddd1]
					dtrain=xgb.DMatrix(data=as.matrix(X.fea0),label=Y1, missing=NA)
					dtest=xgb.DMatrix(data=as.matrix(A[foldsCV[[i]],inddd1]),label=y[foldsCV[[i]]], missing=NA)
					param <- list(objective = "binary:logistic"
							  , subsample = 0.5
							  , max_depth = 1
							  , colsample_bytree = 0.5
							  , eta = 0.01
							  , eval_metric = 'auc'#f1score_eval# map: original
							  , min_child_weight = 1,scale_pos_weight=sum(Y1==0)/sum(Y1==1)) 
					nrounds=1000
					xgb <- xgb.train(params = param, data = dtrain, nrounds = nrounds, verbose = 0,prediction=TRUE,maximize=TRUE)
					temp001 <- predict(xgb,dtest)
					pred[foldsCV[[i]]] <- temp001
					SGene[[sg0]]=xgb.importance(model=xgb)$Feature[1:28]
					sen0[[sg0]]=sum((pred[foldsCV[[i]]]>0.5)==(Y[foldsCV[[i]]]))/length(foldsCV[[i]]);
					sg0=sg0+1
				}
				tos=cbind(y,pred) 
				colnames(tos)=c("Y","Pred") 
				AA=(pred>0.5) 
				acc=sum(AA==Y)/length(Y) #pred: 0.735 for 72 
				prec1=sum((pred>0.5)&(Y==1))/(sum(pred>0.5)+.00001) #prec: 0.733 
				prec0=sum((pred<=0.5)&(Y==0))/(sum(pred<=0.5)+.00001) #prec: 0.736 
				recall1=sum((pred>0.5)&(Y==1))/(sum(Y==1)+.00001) #recall: 0.702 
				recall0=sum((pred<0.5)&(Y==0))/(sum(Y==0)+.00001) #recall: 0.765 
				auc0=pROC::roc(Y,pred,smooth=F) # AUC: 0.813 
				sen=sum((pred>0.5)&(Y==1))/(sum(Y==1)+.00001)# sensitivity
				spec=sum((pred<=0.5)&(Y==0))/(sum(Y==0)+.00001)# specificity
				f1score=(1/(1/prec1+1/recall1)+1/(1/prec0+1/recall0))
				STATT[seed1-2019,]=c(acc,prec1,prec0,recall1,recall0,f1score,as.numeric(auc0$auc),sen,spec) 
			}
			STATT0[kk,]=apply(STATT,2,mean)
		} 
		ii=which.max(STATT0[,1])
		print(STATT0[ii,])
		if((STATT0[ii,1]<ACC0-1e-2)&(length(inddd)<length(selected))){print(inddd);break;}
		if((STATT0[ii,1]>=ACC0-1e-2)|(length(inddd)==length(selected)))
		{
		inddd=inddd[-ii]
		ACC0=STATT0[ii,1]
		}
	}
	toselect=colnames(A0T)[inddd]
	write.csv(toselect,paste0('XGB_toselect_',start,'.csv'),row.names=F,quote=F)
}
##################################################################
A=read.csv('AllT.csv',check.names = FALSE)
L=dim(A)
y=Y=A[,L[2]]
X=A[,-L[2]]
p=dim(X)[2]
for(start in c(0.8,0.9,0.95))
#for(start in c(0.8))
{
	toselect=as.character(read.csv(paste0('XGB_toselect_',start,'.csv'))[,1])	
	X.fea=X
	STATT=matrix(NA,10,7);colnames(STATT)=c("ACC","PREC1","PREC0","RECALL1","RECALL0","F1SCORE","AUC")
	out=paste0('xgb/final_',start,'/')
	system(paste0('mkdir -p ',out,'/out'))
	AUCT=read.csv('AUCT.csv')
	SCORE=matrix(0,50,p)
	colnames(SCORE)=colnames(X.fea)
	PredT=matrix(NA,length(y),10)
	colnames(PredT)=2020:2029
	for(seed1 in 2020:2029)
	{
		print(seed1)
		set.seed(seed1)
		foldsCV <- createFolds(y, k=5, list=TRUE, returnTrain=FALSE)
		pred=y;pred[T]=NA
		for(i in 1:5)
		{
			set.seed(0)
			Y1=y[-foldsCV[[i]]]
			X.fea0=X.fea[-foldsCV[[i]],]
			AUC=AUCT[paste0('X',seed1,'_',i)][,1]
			inddd=which(colnames(X.fea0)%in%toselect)
			X.fea0=X.fea0[,inddd]
			dtrain=xgb.DMatrix(data=as.matrix(X.fea0),label=Y1, missing=NA)
			dtest=xgb.DMatrix(data=as.matrix(X.fea[foldsCV[[i]],inddd]),label=Y[foldsCV[[i]]], missing=NA)
			foldsCV0 <- createFolds(Y1, k=5, list=TRUE, returnTrain=FALSE)
			param <- list(objective = "binary:logistic"
					  , subsample = 0.5
					  , max_depth = 1
					  , colsample_bytree = 0.5
					  , eta = 0.01
					  , eval_metric = 'auc'#f1score_eval# map: original
					  , min_child_weight = 1,scale_pos_weight=sum(Y1==0)/sum(Y1==1)) 
			nrounds=1000
			xgb <- xgb.train(params = param, data = dtrain, nrounds = nrounds, verbose = 0)
			final=as.matrix(xgb.importance(model=xgb))
			aaa=as.numeric(final[,2]);names(aaa)=final[,1]
			SCORE[i+(seed1-2020)*5,names(aaa)]=aaa
			pred[foldsCV[[i]]] <- predict(xgb,dtest)
		}
		tos=cbind(y,pred) 
		colnames(tos)=c("Y","Pred") 
		acc=sum((pred>0.5)==Y)/length(Y) 
		prec1=sum((pred>0.5)&(Y==1))/(sum(pred>0.5)+.00001) 
		prec0=sum((pred<=0.5)&(Y==0))/(sum(pred<=0.5)+.00001) 
		recall1=sum((pred>0.5)&(Y==1))/(sum(Y==1)+.00001) 
		recall0=sum((pred<0.5)&(Y==0))/(sum(Y==0)+.00001) 
		auc0=pROC::roc(Y,pred,smooth=F) 
		f1score=(1/(1/prec1+1/recall1)+1/(1/prec0+1/recall0))
		statt0=c(acc,prec1,prec0,recall1,recall0,f1score,as.numeric(auc0$auc)) 
		STATT[seed1-2019,]=t(round(statt0,4))
		PredT[,seed1-2019]=pred
		print(STATT[seed1-2019,])
	}
	write.csv(PredT,file=paste0(out,'out/PredT.csv'),row.names=F,quote=F)
	write.csv(t(SCORE),file=paste0(out,'out/importanceT.csv'),quote=F)
	write.csv(STATT,file=paste0(out,'out/STATT.csv'),row.names=F,quote=F)
	tempp=sort(apply(SCORE,2,mean),decreasing=T,ind=T) 
	mscore=tempp$x 
	iscore=tempp$ix 
	SCORE1=SCORE[,iscore] 
	write.table(mscore,paste0(out,"/importance.txt"),quote=F)
}
##################################################################
#LightGBM with Fixed feature: Backward deletion					 #
##################################################################
#module load r/3.5.2;R
library(lightgbm)
library(caret)
A0T=read.csv('AllT.csv')
L=dim(A0T)
y=Y=A0T[,L[2]]
A0T=A0T[,-L[2]]
out='LGB/eta0.01W2C0.7S0.5/out'
SCORE=read.csv(paste0(out,'/importanceT.csv'),check.names = FALSE)
rownames(SCORE)=SCORE[,1];SCORE=SCORE[,-1];SCORE=t(SCORE)
tempp=sort(apply(SCORE,2,mean),decreasing=T,ind=T) 
mscore=tempp$x
iscore=tempp$ix
SCORE1=SCORE[,iscore]
cmscore=cumsum(mscore)
#for(start in c(0.8,0.9,0.95))
for(start in c(0.8))
{ 
	cmIND=min(which(cmscore>start)[1],30)
	selected=colnames(SCORE1)[1:cmIND]
	AUCT=read.csv('AUCT.csv')
	l0=dim(AUCT)[1]
	rownames(AUCT)=AUCT[,1]
	AUCT=AUCT[,-1]
	colnames(A0T)=row.names(AUCT)
	ACC0=0
	##################################################################
	A=A0T
	inddd=which(colnames(A0T)%in%selected)
	while(1)
	{
		ll=length(inddd)
		STATT0=matrix(NA,ll,9)
		colnames(STATT0)=c("ACC","PREC1","PREC0","RECALL1","RECALL0","f1score","AUC",'Sensitivity','Specificity')
		for(kk in 1:ll)
		{		
			print(ll-kk)
			STATT=matrix(NA,10,9);
			colnames(STATT)=c("ACC","PREC1","PREC0","RECALL1","RECALL0","f1score","AUC",'Sensitivity','Specificity')
			SGene=NULL;sg0=1;sen0=NULL
			for(seed1 in 2020:2029)
			{
				set.seed(seed1)
				foldsCV <- createFolds(y, k=5, list=TRUE, returnTrain=FALSE)
				########################################################
				p=dim(A)[2]
				pred=y;pred[T]=NA
				for(i in 1:5)
				{
					set.seed(0)
					Y1=y[-foldsCV[[i]]]
					X.fea0=A[-foldsCV[[i]],]
					p=dim(X.fea0)[2]
					X.fea00=X.fea0
					AUC=AUCT[paste0('X',seed1,'_',i)][,1]
					inddd1= inddd[-kk]
					X.fea0=X.fea00[,inddd1]
					dtrain=lgb.Dataset(as.matrix(X.fea0), label=Y1,missing=NA)
					param = list(objective = "binary",metric = "auc",
					min_sum_hessian_in_leaf = 2,feature_fraction = 0.7,learning_rate = 0.01, max_depth=1,
					bagging_fraction = 0.5,is_unbalance = T)
					nrounds=1000
					lgb <-lgb.train(params = param, data = dtrain,nrounds = nrounds,verbose=-1)
					temp001 <- predict(lgb,as.matrix(A[foldsCV[[i]],inddd1]))
					pred[foldsCV[[i]]] <- temp001
				}
				tos=cbind(y,pred) 
				colnames(tos)=c("Y","Pred") 
				AA=(pred>0.5) 
				acc=sum(AA==Y)/length(Y) #pred: 0.735 for 72 
				prec1=sum((pred>0.5)&(Y==1))/(sum(pred>0.5)+.00001) #prec: 0.733 
				prec0=sum((pred<=0.5)&(Y==0))/(sum(pred<=0.5)+.00001) #prec: 0.736 
				recall1=sum((pred>0.5)&(Y==1))/(sum(Y==1)+.00001) #recall: 0.702 
				recall0=sum((pred<0.5)&(Y==0))/(sum(Y==0)+.00001) #recall: 0.765 
				auc0=pROC::roc(Y,pred,smooth=F) # AUC: 0.813 
				sen=sum((pred>0.5)&(Y==1))/(sum(Y==1)+.00001)# sensitivity
				spec=sum((pred<=0.5)&(Y==0))/(sum(Y==0)+.00001)# specificity
				f1score=(1/(1/prec1+1/recall1)+1/(1/prec0+1/recall0))
				STATT[seed1-2019,]=c(acc,prec1,prec0,recall1,recall0,f1score,as.numeric(auc0$auc),sen,spec) 
			}
			STATT0[kk,]=apply(STATT,2,mean)
		} 
		ii=which.max(STATT0[,1])
		print(STATT0[ii,])
		if((STATT0[ii,1]<ACC0-1e-2)&(length(inddd)<length(selected))){print(inddd);break;}
		if((STATT0[ii,1]>=ACC0-1e-2)|(length(inddd)==length(selected)))
		{
		inddd=inddd[-ii]
		ACC0=STATT0[ii,1]
		}
	}
	toselect=colnames(A0T)[inddd]
	write.csv(toselect,paste0('LGB_toselect_',start,'.csv'),row.names=F,quote=F)
}
[1] "Trabe+Energy"                        "Demo*Clinic+Gender*MusSor"          
[3] "Serum*Saliva+VE-cad_Ser*ANG_Sal"     "Saliva*Clinic+PAI-1_Sal*RangeWOpain"
##################################################################
A=read.csv('AllT.csv',check.names = FALSE)
L=dim(A)
y=Y=A[,L[2]]
X=A[,-L[2]]
p=dim(X)[2]
#for(start in c(0.8,0.9,0.95))
for(start in c(0.8))
{
	toselect=as.character(read.csv(paste0('LGB_toselect_',start,'.csv'))[,1])	
	X.fea=X
	STATT=matrix(NA,10,7);colnames(STATT)=c("ACC","PREC1","PREC0","RECALL1","RECALL0","F1SCORE","AUC")
	out=paste0('LGB/final_',start,'/')
	system(paste0('mkdir -p ',out,'/out'))
	AUCT=read.csv('AUCT.csv')
	SCORE=matrix(0,50,p)
	colnames(SCORE)=colnames(X.fea)
	PredT=matrix(NA,length(y),10)
	colnames(PredT)=2020:2029
	for(seed1 in 2020:2029)
	{
		print(seed1)
		set.seed(seed1)
		foldsCV <- createFolds(y, k=5, list=TRUE, returnTrain=FALSE)
		pred=y;pred[T]=NA
		for(i in 1:5)
		{
			set.seed(0)
			Y1=y[-foldsCV[[i]]]
			X.fea0=X.fea[-foldsCV[[i]],]
			AUC=AUCT[paste0('X',seed1,'_',i)][,1]
			inddd=which(colnames(X.fea0)%in%toselect)
			X.fea0=X.fea0[,inddd]
			dtrain=lgb.Dataset(as.matrix(X.fea0), label=Y1,missing=NA)
			param = list(objective = "binary",metric = "auc",
			min_sum_hessian_in_leaf = 2,feature_fraction = 0.7,learning_rate = 0.01, max_depth=1,
			bagging_fraction = 0.5,is_unbalance = T)
			nrounds=1000
			lgb <-lgb.train(params = param, data = dtrain,nrounds = nrounds,verbose=-1)
			final=as.matrix(lgb.importance(model=lgb))
			aaa=as.numeric(final[,2]);names(aaa)=final[,1]
			SCORE[i+(seed1-2020)*5,names(aaa)]=aaa
			pred[foldsCV[[i]]] <- predict(lgb,as.matrix(X.fea[foldsCV[[i]],inddd]))
		}
		tos=cbind(y,pred) 
		colnames(tos)=c("Y","Pred") 
		acc=sum((pred>0.5)==Y)/length(Y) 
		prec1=sum((pred>0.5)&(Y==1))/(sum(pred>0.5)+.00001) 
		prec0=sum((pred<=0.5)&(Y==0))/(sum(pred<=0.5)+.00001) 
		recall1=sum((pred>0.5)&(Y==1))/(sum(Y==1)+.00001) 
		recall0=sum((pred<0.5)&(Y==0))/(sum(Y==0)+.00001) 
		auc0=pROC::roc(Y,pred,smooth=F) 
		f1score=(1/(1/prec1+1/recall1)+1/(1/prec0+1/recall0))
		statt0=c(acc,prec1,prec0,recall1,recall0,f1score,as.numeric(auc0$auc)) 
		STATT[seed1-2019,]=t(round(statt0,4))
		PredT[,seed1-2019]=pred
		print(STATT[seed1-2019,])
	}
	write.csv(PredT,file=paste0(out,'out/PredT.csv'),row.names=F,quote=F)
	write.csv(t(SCORE),file=paste0(out,'out/importanceT.csv'),quote=F)
	write.csv(STATT,file=paste0(out,'out/STATT.csv'),row.names=F,quote=F)
	tempp=sort(apply(SCORE,2,mean),decreasing=T,ind=T) 
	mscore=tempp$x 
	iscore=tempp$ix 
	SCORE1=SCORE[,iscore] 
	write.table(mscore,paste0(out,"/importance.txt"),quote=F)
}


