# ---- init ----

library(caret)
library(dplyr)

set.seed(1337)

# ---- load-data ----

ds.train = read.csv("pml-training.csv", strip.white=TRUE)
ds.test  = read.csv("pml-testing.csv",  strip.white=TRUE)

# ---- preprocess ----

completeColumnsId = which(!apply(is.na(ds.train), 2, any) == TRUE)
divZeroColumnsId  = which(!apply(ds.train == "#DIV/0!", 2, any, na.rm=TRUE) == TRUE)

ds.train.sub = select(ds.train, intersect(completeColumnsId, divZeroColumnsId))
ds.test.sub  = select(ds.test,  intersect(completeColumnsId, divZeroColumnsId))

ds.train.pred = select(ds.train.sub, roll_belt:classe)
ds.test.pred  = select(ds.test.sub,  roll_belt:magnet_forearm_z)

# ---- build-model ----

inTrain = createDataPartition(y=ds.train.pred$classe,
                              p=0.6, list=FALSE)

ds.valid.pred = filter(ds.train.pred, !(row_number() %in% inTrain))
ds.train.pred = filter(ds.train.pred, row_number() %in% inTrain)

ctrl = trainControl(method="repeatedcv", number=10, repeats=3)

knnFit = train(classe ~ ., data=ds.train.pred, method="knn",
                trControl=ctrl, metric="Accuracy", tuneLength=5,
                preProc=c("center", "scale"))

knnPredict = predict(knnFit, newdata=ds.valid.pred)
cmat = confusionMatrix(knnPredict, ds.valid.pred$classe)

# ---- predict-outcomes ----

answers = predict(knnFit, newdata=ds.test.pred)

# ---- save-outputs ----

pml_write_files = function(x)
{
    n = length(x)
    for(i in 1:n){
        filename = paste0("problem_id_",i,".txt")
        write.table(x[i],file=filename,quote=FALSE,row.names=FALSE,col.names=FALSE)
    }
}

pml_write_files(answers)