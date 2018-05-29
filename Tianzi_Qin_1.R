## connect database and library packages
library(DBI)
library(RMySQL)
myhost <- "localhost"
mydb <- "studb"
myacct <- "cis434"
mypwd <- "LLhtFPbdwiJans8F@S207"
driver <- dbDriver("MySQL")
conn <- dbConnect(driver, host=myhost, dbname=mydb,
                  myacct, mypwd)
classification <- dbGetQuery(conn, "SELECT * FROM classification WHERE rtag='jJ+y1j03+Xbe'")
dbDisconnect(conn) 

library(tm)
library(e1071)
library(maxent)
library(pROC)   

## Load train dataset which was about sentiment of airline related tweets
tweet<-read.csv("Tweets.csv")
complaint <-subset(tweet[c("tweet_id","airline_sentiment","text")],tweet$airline_sentiment=="negative")
noncomplaint <- subset(tweet[c("tweet_id","airline_sentiment","text")],tweet$airline_sentiment=="positive"|tweet$airline_sentiment=="neutral")

## Resample the original dataset, use all negative tweets as complaint sample and some positive and neutral tweets(15% of #negative tweets) as noncomplaint sample
set.seed(1) 
a<-sample(1:nrow(noncomplaint),nrow(complaint)*0.15)
noncomplaint1<-noncomplaint[a,]
temp<-merge(complaint,noncomplaint1,all=TRUE)
temp$airline_sentiment<-as.character(temp$airline_sentiment)
temp$airline_sentiment[temp$airline_sentiment=="negative"]<-"complaint"
temp$airline_sentiment[temp$airline_sentiment=="positive"|temp$airline_sentiment=="neutral"]<-"noncomplaint"
temp$airline_sentiment<-as.factor(temp$airline_sentiment)
rm(noncomplaint)
Y<-temp$airline_sentiment
temp1<-temp[c(1,3)]
names(temp1) = c("doc_id", "text") 

## Build dtm and use frequency of words as X features in machine learning models
docs <- Corpus(DataframeSource(temp1))
stop1<-c("@","airline","jetblue",as.character(unique(tweet$airline),"virginamerica"))
dtm.control = list(tolower=T, removePunctuation=T, removeNumbers=T, stopwords=c(stopwords("english"),stop1), stripWhitespace=T, stemming=F)
dtm.full <- DocumentTermMatrix(docs, control=dtm.control)
dtm <- removeSparseTerms(dtm.full,0.995)
X <- as.matrix(dtm) 

## Divide 75% as training data,25% as validation data
set.seed(1) 
n=length(Y)
n1=round(n*0.75)
n2=n-n1
train=sample(1:n,n1) 

## Build dtm for test data
temp2 <- classification[,c("id","tweet")]
names(temp2) = c("doc_id", "text") 
docs <- Corpus(DataframeSource(temp2)) 
stop2<-c("@","airline",unique(classification$airline))
dtm.control = list(tolower=T, removePunctuation=T, removeNumbers=T, stopwords=c(stopwords("english"),stop2), stripWhitespace=T, stemming=F)
dtm.full <- DocumentTermMatrix(docs, control=dtm.control)
dtm <- removeSparseTerms(dtm.full,0.995)
X1 <- as.matrix(dtm) 

## Set function for f1 score
Evaluation <- function(pred, true, class)
{
  tp <- sum( pred==class & true==class)
  fp <- sum( pred==class & true!=class)
  tn <- sum( pred!=class & true!=class)
  fn <- sum( pred!=class & true==class)
  precision <- tp/(tp+fp)
  recall <- tp/(tp+fn)
  f1<-(1/2*1/precision+1/2*1/recall) ^-1
  print(f1)
}

## Naive Bayes prediction
nb.model <- naiveBayes(X[train,], Y[train])
pred.class <- predict( nb.model,X[-train,])
table( pred.class, Y[-train] )
Evaluation( pred.class, Y[-train], "noncomplaint" ) #F1:0.351
## Maximum Entropy prediction
maxent.model <- maxent(X[train,],Y[train])
pred <- predict( maxent.model, X[-train,])
pred.class<-pred[,1]
table( pred.class, Y[-train] )
Evaluation( pred.class, Y[-train], "noncomplaint" ) #F1:0.528
roc <- roc(as.numeric(Y[-train]), as.numeric(as.factor(pred.class)))
plot.roc(roc)
## SVM prediction
X2<-X[,colnames(X)%in%colnames(X1)]
svm.model <- svm(Y[train]~ ., X2[train,])
pred <- predict( svm.model, X2[-train,] )
table(pred, Y[-train])
Evaluation( pred, Y[-train], "noncomplaint" ) #F1:0.230

## Use the model with highest f1 score as the final model for test dataset
pred <- predict( maxent.model, X1 )
p1<-classification$tweet[pred=="noncomplaint"]
p2<-classification$id[pred=="noncomplaint"]

## Check the prediction result 
p1<-p1[-c(2:4,6,10,13,18,20,40,47:49,51,54:56,63,68,72:74,76,79,82:84,102,105:107,111,115,123,125,128,131,132,137:139,152,156,157,159,162,165,167,169,170,172,173,175:177,181,185,186,189,192)]
p2<-p2[-c(2:4,6,10,13,18,20,40,47:49,51,54:56,63,68,72:74,76,79,82:84,102,105:107,111,115,123,125,128,131,132,137:139,152,156,157,159,162,165,167,169,170,172,173,175:177,181,185,186,189,192)]
df<-data.frame(id=p2,tweet=p1)
write.csv(df,file="noncomplaint.csv")
                     