#Code from http://rstudio-pubs-static.s3.amazonaws.com/265200_a8d21a65d3d34b979c5aafb0de10c221.html
library(plyr)
db.adult <- read.table("adult.data", sep=",", header=FALSE, na.strings = " ?")
colnames(db.adult) <- c("age", "workclass", "fnlwgt", "education", 
                        "education_num", "marital_status", "occupation",
                        "relationship", "race", "sex", "capital_gain", 
                        "capital_loss", "hours_per_week", "native_country", "income");
db.adult <- na.omit(db.adult);

db.adult$hours_w[db.adult$hours_per_week < 40] <- " less_than_40"
db.adult$hours_w[db.adult$hours_per_week >= 40 & 
                 db.adult$hours_per_week <= 45] <- " between_40_and_45"
db.adult$hours_w[db.adult$hours_per_week > 45 &
                 db.adult$hours_per_week <= 60  ] <- " between_45_and_60"
db.adult$hours_w[db.adult$hours_per_week > 60 &
                 db.adult$hours_per_week <= 80  ] <- " between_60_and_80"
db.adult$hours_w[db.adult$hours_per_week > 80] <- " more_than_80"
db.adult$hours_w <- factor(db.adult$hours_w,
                           ordered = FALSE,
                           levels = c(" less_than_40", 
                                      " between_40_and_45", 
                                      " between_45_and_60",
                                      " between_60_and_80",
                                      " more_than_80"));

Asia_East <- c(" Cambodia", " China", " Hong", " Laos", " Thailand",
               " Japan", " Taiwan", " Vietnam")

Asia_Central <- c(" India", " Iran")

Central_America <- c(" Cuba", " Guatemala", " Jamaica", " Nicaragua", 
                     " Puerto-Rico",  " Dominican-Republic", " El-Salvador", 
                     " Haiti", " Honduras", " Mexico", " Trinadad&Tobago")

South_America <- c(" Ecuador", " Peru", " Columbia")


Europe_West <- c(" England", " Germany", " Holand-Netherlands", " Ireland", 
                 " France", " Greece", " Italy", " Portugal", " Scotland")

Europe_East <- c(" Poland", " Yugoslavia", " Hungary")

db.adult <- mutate(db.adult, 
       native_region = ifelse(native_country %in% Asia_East, " East-Asia",
                ifelse(native_country %in% Asia_Central, " Central-Asia",
                ifelse(native_country %in% Central_America, " Central-America",
                ifelse(native_country %in% South_America, " South-America",
                ifelse(native_country %in% Europe_West, " Europe-West",
                ifelse(native_country %in% Europe_East, " Europe-East",
                ifelse(native_country == " United-States", " United-States", 
                       " Outlying-US" ))))))))
db.adult$native_region <- factor(db.adult$native_region, ordered = FALSE)
db.adult <- mutate(db.adult, 
            cap_gain = ifelse(db.adult$capital_gain < 3464, " Low",
                       ifelse(db.adult$capital_gain >= 3464 & 
                              db.adult$capital_gain <= 14080, " Medium", " High")))


db.adult$cap_gain <- factor(db.adult$cap_gain,
                            ordered = TRUE,
                            levels = c(" Low", " Medium", " High"))
db.adult <- mutate(db.adult, 
            cap_loss = ifelse(db.adult$capital_loss < 1672, " Low",
                       ifelse(db.adult$capital_loss >= 1672 & 
                              db.adult$capital_loss <= 1977, " Medium", " High")))


db.adult$cap_loss <- factor(db.adult$cap_loss,
                            ordered = TRUE,
                            levels = c(" Low", " Medium", " High"))

db.adult$workclass <- droplevels(db.adult$workclass)

levels(db.adult$workclass)

db.test <- read.table("adult.test",
                      sep = ",", 
                      header = FALSE, 
                      skip = 1, 
                      na.strings = " ?")

colnames(db.test) <- c("age", "workclass", "fnlwgt", "education",
                       "education_num", "marital_status", "occupation",
                       "relationship", "race", "sex", "capital_gain",
                       "capital_loss", "hours_per_week",
                       "native_country", "income")

db.test <- na.omit(db.test)

row.names(db.test) <- 1:nrow(db.test)

levels(db.test$income)[1] <- " <=50K"
levels(db.test$income)[2] <- " >50K"

db.test$hours_w[db.test$hours_per_week < 40] <- " less_than_40"
db.test$hours_w[db.test$hours_per_week >= 40 & 
                db.test$hours_per_week <= 45] <- " between_40_and_45"
db.test$hours_w[db.test$hours_per_week > 45 &
                db.test$hours_per_week <= 60  ] <- " between_45_and_60"
db.test$hours_w[db.test$hours_per_week > 60 &
                db.test$hours_per_week <= 80  ] <- " between_60_and_80"
db.test$hours_w[db.test$hours_per_week > 80] <- " more_than_80"



db.test$hours_w <- factor(db.test$hours_w,
                          ordered = FALSE,
                          levels = c(" less_than_40", 
                                     " between_40_and_45", 
                                     " between_45_and_60",
                                     " between_60_and_80",
                                     " more_than_80"))

db.test <- mutate(db.test, 
       native_region = ifelse(native_country %in% Asia_East, " East-Asia",
                ifelse(native_country %in% Asia_Central, " Central-Asia",
                ifelse(native_country %in% Central_America, " Central-America",
                ifelse(native_country %in% South_America, " South-America",
                ifelse(native_country %in% Europe_West, " Europe-West",
                ifelse(native_country %in% Europe_East, " Europe-East",
                ifelse(native_country == " United-States", " United-States", 
                       " Outlying-US" ))))))))


db.test$native_region <- factor(db.test$native_region, ordered = FALSE)
db.test <- mutate(db.test, 
            cap_gain = ifelse(db.test$capital_gain < 3464, " Low",
                       ifelse(db.test$capital_gain >= 3464 & 
                              db.test$capital_gain <= 14080, " Medium", " High")))

db.test$cap_gain <- factor(db.test$cap_gain,
                            ordered = FALSE,
                            levels = c(" Low", " Medium", " High"))
db.test<- mutate(db.test, 
            cap_loss = ifelse(db.test$capital_loss < 1672, " Low",
                       ifelse(db.test$capital_loss >= 1672 & 
                              db.test$capital_loss <= 1977, " Medium", " High")))


db.test$cap_loss <- factor(db.test$cap_loss,
                            ordered = FALSE,
                            levels = c(" Low", " Medium", " High"))
db.test$workclass <- droplevels(db.test$workclass)
write.csv(db.adult, "adult_df.csv", row.names = FALSE)

write.csv(db.test, "test_df.csv", row.names = FALSE)

#adult <- read.csv("adult_df.csv")
#test <- read.csv("test_df.csv")
