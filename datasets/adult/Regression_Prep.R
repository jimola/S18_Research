library(plyr)
library(compiler);
library(foreign);
library(pROC);
library(stringr);

create_dummies <- function(row, name){
	l <- levels(row)
	return(list(names=paste(name, as.character(2:length(l)), sep="_"),
		data=lapply(l[2:length(l)], function(c){return(as.integer(row == c))})))
}
get_country_indicator <- function(d, countries){
	l <- integer(length(d$native.country))
	for(c in countries){
		l <- l + as.integer(d$native.country == c)
	}
	return(l)
}

prepare_database_adult <- function(d){
	Asia_East <- get_country_indicator(d, c(" Cambodia", " China", " Hong", " Laos", " Thailand",
               " Japan", " Taiwan", " Vietnam"))
	Asia_Central <- get_country_indicator(d, c(" India", " Iran"))
	Central_America <- get_country_indicator(d, c(" Cuba", " Guatemala", " Jamaica", " Nicaragua", 
                     " Puerto-Rico",  " Dominican-Republic", " El-Salvador", 
                     " Haiti", " Honduras", " Mexico", " Trinadad&Tobago"))
	South_America <- get_country_indicator(d, c(" Ecuador", " Peru", " Columbia"))
	Europe_West <- get_country_indicator(d, c(" England", " Germany", " Holand-Netherlands", " Ireland", 
                 " France", " Greece", " Italy", " Portugal", " Scotland"))
	Europe_East <- get_country_indicator(d, c(" Poland", " Yugoslavia", " Hungary"))
	
	work_class <- create_dummies(d$workclass, 'wkclass')
	edu <- create_dummies(d$education, 'edu')
	marital <- create_dummies(d$marital.status, 'marital')
	occ <- create_dummies(d$occupation, 'occ')
	race <- create_dummies(d$race, 'race')
	sex <- create_dummies(d$sex, 'sex')
	earn <- create_dummies(d$earning, 'earn')

	df<-data.frame(d$age, work_class$data, edu$data, d$education.num,
						Asia_East, Asia_Central, Central_America,
						South_America, Europe_East, Europe_West,
						marital$data, occ$data, race$data, sex$data, d$capital.gain,
						d$capital.loss, d$hours.per.week, earn$data)
	names <- c("age", work_class$names, edu$names, "edu_num", 
				"asia_e", "asia_c", "c_america", "s_america",
				"europe_e", "europe_w", marital$names, occ$names,
				race$names, sex$names, "capital_gain", "capital_loss",
				"hrs_per_week", "ys")
	names(df) <- names
	return(df)
}

#diffpLR.reverse.xi.bias <- function(data, class, guess, epsilon, R, lambda)
#diffpLR.reverse.xi <- function(data, class, guess, epsilon, R, lambda)
#diffpLR.reverse.chaudhuri <- function(data, class, guess, epsilon, R, lambda)
#diffpLR.reverse.functional <- function(data, class, epsilon, guess)


#diffpLR.xi.tuned <- function(train, class, guess=vkoattr, epsilon=1.0, trainsize=105, testsize=124, params.R=c(0.25,0.5,1), params.lambda=c(0.002,0.004,0.008,0.016,0.032,0.064,0.128,0.256))
#diffpLR.chaudhuri.tuned <- function(train, class, guess=vkoattr, epsilon=1.0, trainsize=105, testsize=124, params.R=c(0.25,0.5,1), params.lambda=c(0.002,0.004,0.008,0.016,0.032,0.064,0.128,0.256))
df <- prepare_database(read.csv(
	'../data-unsorted/adult/adult_cleaner_no_relationship_normalized.csv'))

c <- as.integer(0.8*nrow(df))
df_train <- df[0:(c-1),]
df_test <- df[c:nrow(df),]
get_error <- function(e){e1 <- e - df_test$ys; return(mean(e1*e1))}
get_error_sharp <- function(e){x <- as.integer(e > 0.5); x <- x - df_test$ys; return(mean(x*x))}
get_levels <- function(epsilon){
	xi_bias <- diffpLR.reverse.xi.bias(df_train, "ys", list(), epsilon, 1, 0.25)$forward
	xi <- diffpLR.reverse.xi(df_train, "ys", list(), epsilon, 1, 0.25)$forward
	cha <- diffpLR.reverse.chaudhuri(df_train, "ys", list(), epsilon, 1, 0.25)$forward
	fun <- diffpLR.reverse.functional(df_train, "ys", epsilon, list())$forward

	preds <- matrix(c(apply(df_test, 1, xi), apply(df_test, 1, xi_bias), apply(df_test, 1, cha), apply(df_test, 1, fun)), ncol=4)
	errs <- apply(preds, 2, get_error)
	return(data.frame(err=errs, eps=epsilon, type=c('xi', 'xib', 'cha', 'fun')))
}

get_errors <- function(eps){
	errs <- data.frame()
	for(x in eps){
		errs <- rbind(get_levels(x), errs)
	}
	return(errs)
}

eps <- c(0.1, 0.2, 0.4, 0.8, 1.6, 3.2, 10, 100)

ggplot(data=errs[errs$type != 'fun',], aes(x=log(eps), y=err, group=type)) + geom_line() + geom_point()