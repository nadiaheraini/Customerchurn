##############################################
### Program Klasifikasi Customer Churn  ######
### Created By Nadia Ananda Heraini ##########
### Metode Klasifikasi: Naive Bayes ##########
##############################################
# Setting lokasi data
setwd("D:/FDC/")

# Setting Library
library(e1071)
library(caret)
library(readxl)
library(openxlsx)
library(data.table)
library(summarytools)
library(dplyr)
library(ROCR)

###################
# Aktifkan function
###################
isi_data_missing=function(data_df){
  data1=data_df
  lstcont = c()
  for (i in names(data1)){
    if (typeof(data1[[i]]) == "character"){
      data1[[i]] = scorecard::replace_na(data1[[i]], 'Missing')
    } 
    else{
      if (i != 'y'){
        lstcont = append(lstcont, i) 
      }
      data1[[i]] = scorecard::replace_na(data1[[i]], -999)
    }
  }
  return(data1)
}
#####################

# Import dataset dari format CSV:  
data_training=read.csv("findata_challenge_train.csv")
data_testing=read.csv("findata_challenge_test.csv")

# Cek dataset: komposisi dan data missing  
#1.data_training
summarytools::dfSummary(data_training)
#2.data_testing
summarytools::dfSummary(data_testing)
# Hasil rerata data mising tidak melebihi dari 1% sehingga perlu adjustment 

#### Proses Data Blank/"" pada type character diubah menjadi NA
#1. data_training
temp_data_training <-data_training
temp_data_training[temp_data_training ==""] <-NA # isi dengan NA
data_training<-temp_data_training

#2. data_testing 
temp_data_testing <-data_testing
temp_data_testing[temp_data_testing ==""] <-NA # isi dengan NA
data_testing<-temp_data_testing

######## Proses adjustment data missing
# Cari nilai minimum pada seluruh data numerik didapat - ratusan = -999 
# Melengkapi data missing value, jika numerik isi ='-999', jika character='Missing'
# Dataset: data_training dan data_testing

###### Isi data missing dengan -999 atau "Missing"  
#1. data_training
data_training=isi_data_missing(data_training)
#2. data_testing
data_testing=isi_data_missing(data_testing)

## konversi struktur data pada y menjadi factor untuk membuat model
data_training$y= as.factor(data_training$y)

###########################
### Model Fitting #########
###########################
# Membuat model dengan data_training
# metode klasifikasi yang digunakan menggunakan naiveBayes
model=naiveBayes(x=data_training %>% select(-y),
                 y=data_training$y)   
## Proses Prediksi data_testing dengan menggunakan model dari data_training  
## Untuk bisa dihitung Accuracynya maka data_testing dianggap data baru
##  Hasil prediksinya dimasukan sebagai variabel y predicted

# Proses Prediksi pada data_testing untuk mendapatkan y predicted
prediksi_data_testing=predict(model,data_testing)
# simpan hasil prediksi
hasil_prediksi_data_testing=data.frame(prediksi_data_testing)
#simpan hasil prediksi ke CSV
write.csv(hasil_prediksi_data_testing,"nadia_ananda_heraini-submission.csv")
# Proses penggabungan data_testing setelah mendapatkan y predicted
gabung_data_testing <- data.frame(data_testing,hasil_prediksi_data_testing)

##################################################
### Evaluasi Model untuk melihat performance model
### 1. Hitung prediksi (accuracy model) ##########
### 2. Hitung nilai ROC/AUC ######################
##################################################
# persiapan data
data_test=gabung_data_testing # data_test, data baru (hasil penggabungan data)
levels(data_test$prediksi_data_testing) # cek level
####################
# 1. Hitung Accuracy
####################
# hitung prediski
naive_prob <-predict(model,data_test,type = "raw")
# proses
naive_table <- tibble(y = data_test$prediksi_data_testing)
naive_table$naive_prob_no <- round(naive_prob[,0],5)
naive_table$naive_prob_yes <- round(naive_prob[,1],5)
naive_table$naive_class <- factor(ifelse(naive_prob[,2] > 0.5, 1,0))

cm_naive <- confusionMatrix(naive_table$naive_class, naive_table$y, positive = "0")
# lihat hasil
cm_naive # hasil accuracynya= 1 (100%), model dapat meprediksi sangat baik dan sempurna

eval_naive <- tibble(Accuracy = cm_naive$overall[1],
                     Recall = cm_naive$byClass[1],
                     Specificity = cm_naive$byClass[2],
                     Precision = cm_naive$byClass[3])
# cetak hasil evaluasi 
eval_naive
# nilai accuracynya:1(100%) menunjukkan performance model sangat baik
#> eval_naive
## A tibble: 1 x 4
#Accuracy Recall Specificity Precision
#   <dbl>  <dbl>       <dbl>     <dbl>
#       1        1      1           1         

#####################
# 2. Hitung ROC/AUC
#####################
naive_roc <- data.frame(prediction = naive_table$naive_prob_yes,
                        trueclass = as.numeric(naive_table$y==0))
naive_roc <- prediction(naive_roc$prediction,naive_roc$trueclass) 

# Hasil visual ROC curve dengan nilai AUC= 100% menunjukan model sangat baik dan sempurna
plot(performance(naive_roc, "tpr", "fpr"),main = "ROC",col="red",lty=2)
abline(a = 0, b = 1)

# hitung nilai AUC 
auc_ROCR_n <- performance(naive_roc, measure = "auc")
auc_ROCR_n <- auc_ROCR_n@y.values[[1]]
auc_ROCR_n
# > auc_ROCR_n
#[1] 1
# nilai AUC = 1(100%)

###################################################################
### Membandingkan customer churn pada data_training dan data_testing
####################################################################
table(data_training$y)
# > table(data_training$y)
# 0     1 
# 83008 16992 
table(gabung_data_testing$prediksi_data_testing)
#> table(gabung_data_testing$prediksi_data_testing)
# 0     1 
# 18723  6277 
######################## End of Program #######################################

###############################################
###### Kesimpulan #############################
###############################################
# Kesimpulan:
# Setelah dilakukan kalibrasi atau test performance dengan menggunakan data_testing maka disimpulkan:
# 1. Performance model sangat baik dalam memprediksi dan sempurna, karena:
#    a. Nilai accuracy: 1(100%), sensitifity: 1(100%) dan Specificity: 1=(100%)
#       menunjukkan prediksi model sangat baik
#    b. Nilai AUC/ROC = 1(100%) menunjukkan performance model sangat sempurna.
#        > auc_ROCR_n
#        [1] 1
# 
#       > cm_naive <- confusionMatrix(naive_table$naive_class, naive_table$y, positive = "0")
#       > # lihat hasil
#       > cm_naive # hasil accuracynya= 1 (100%), model dapat meprediksi sangat baik dan sempurna
#       Confusion Matrix and Statistics
#                 Reference
#       Prediction     0     1
#                0 18723     0
#                1     0  6277
#
#                      Accuracy : 1          
#                        95% CI : (0.9999, 1)
#           No Information Rate : 0.7489     
#           P-Value [Acc > NIR] : < 2.2e-16  
#
#                         Kappa : 1          
#
#        Mcnemar's Test P-Value : NA         
#                                     
#                   Sensitivity : 1.0000     
#                   Specificity : 1.0000     
#                Pos Pred Value : 1.0000     
#                Neg Pred Value : 1.0000     
#                    Prevalence : 0.7489     
#                Detection Rate : 0.7489     
#          Detection Prevalence : 0.7489     
#             Balanced Accuracy : 1.0000     
#                                     
#              'Positive' Class : 0          
#> 
# 2. Customer Churn pada data_training dan data_testing:
# > table(data_training$y) # 1000000 observasi # data_training
#    0     1 
#    83008 16992
#    Customer Churn(1) = (16992/(83008+16992))=16.99%
#    Customer No Churn(0)= (83008/(83008+16992))=83.01%
#
# > table(gabung_data_testing$prediksi_data_testing) # 125000 observasi
#    0     1 
#    18723  6277 
#    Customer Churn(1) = (6277/(18723+6277))=25.11%
#    Customer No Churn(0)= (18723/((18723+6277))=74.89.%
############################################################
