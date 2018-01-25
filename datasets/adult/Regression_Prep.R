library(plyr)
library(compiler);
library(foreign);
library(pROC);
library(stringr);
library(ggplot2);
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
normalize_df <- function(df){
	M <- as.matrix(df[,(1:length(df)-1)])
	max_norms <- sqrt(max(rowSums(M*M)))
	d <- data.frame(M/max_norms, df$ys)
	names(d) <- names(df)
	return(d)
}

#diffpLR.xi.tuned <- function(train, class, guess=vkoattr, epsilon=1.0, trainsize=105, testsize=124, params.R=c(0.25,0.5,1), params.lambda=c(0.002,0.004,0.008,0.016,0.032,0.064,0.128,0.256))
#diffpLR.chaudhuri.tuned <- function(train, class, guess=vkoattr, epsilon=1.0, trainsize=105, testsize=124, params.R=c(0.25,0.5,1), params.lambda=c(0.002,0.004,0.008,0.016,0.032,0.064,0.128,0.256))
df <- prepare_database_adult(read.csv(
	'../data-unsorted/adult/adult_cleaner_no_relationship_normalized.csv'))
err_func <- function(e, ys){e1 <- e - ys; return(mean(e1*e1))}
get_levels <- function(epsilon, df_train, df_test){
	xi_bias <- diffpLR.reverse.xi.bias(df_train, "ys", list(), epsilon, 1, 0.25)$forward
	xi <- diffpLR.reverse.xi(df_train, "ys", list(), epsilon, 1, 0.25)$forward
	cha <- diffpLR.reverse.chaudhuri(df_train, "ys", list(), epsilon, 1, 0.25)$forward
	fun <- diffpLR.reverse.functional(df_train, "ys", epsilon, list())$forward
	ys <- df_test$ys
	errs <- c(err_func(apply(df_test, 1, xi), ys), err_func(apply(df_test, 1, xi_bias), ys), 
		err_func(apply(df_test, 1, cha), ys), err_func(apply(df_test, 1, fun), ys))
	return(data.frame(err=errs, eps=epsilon, type=c('xi', 'xib', 'cha', 'fun')))
}
xi_bias_fix <- function(e){
	return(diffpLR.reverse.xi.bias(df_train, "ys", list(), e, 1, 0.25)$forward)
}
xi_fix <- function(e){
	return(diffpLR.reverse.xi(df_train, "ys", list(), e, 1, 0.25)$forward)
}
cha_fix <- function(e){
	return(diffpLR.reverse.chaudhuri(df_train, "ys", list(), e, 1, 0.25)$forward)
}
fun_fix <- function(e){
	return(diffpLR.reverse.functional(df_train, "ys", e, list())$forward)
}

get_errors <- function(df, eps, iters=10, ssize=1000){
	errs <- data.frame()
	for(a in 1:iters){
		ds <- df[sample(nrow(df), ssize),]
		c <- as.integer(0.8*nrow(ds))
		ds_train <- ds[0:(c-1),]
		ds_test <- ds[c:nrow(ds),]
		for(x in eps){
			errs <- rbind(get_levels(x, ds_train, ds_test), errs)
		}
	}
	return(errs)
}

eps <- c(0.1, 0.2, 0.4, 0.8, 1.6, 3.2, 10, 100)

#ggplot(data=errs, aes(x=log(eps), y=err, group=type, color=type)) + geom_line() + geom_point()
search_for_privacy <- function(f, acc){
	eps <- 0.001
	while(!acc(f)(eps)){
		eps <- 2*eps
	}
}
get_info <- function(info, name, eps){
	info <- info[info$type == name,]
	ans <- data.frame()
	for(e in eps){
		m <- as.vector(info[info$eps == e,'err'])
		ans <- rbind(ans, data.frame(eps=e, mean=mean(m), sd=sd(m), type=name))

	}
	return(ans)
}

multiplot <- function(..., plotlist=NULL, file, cols=1, layout=NULL) {
  library(grid)

  # Make a list from the ... arguments and plotlist
  plots <- c(list(...), plotlist)

  numPlots = length(plots)

  # If layout is NULL, then use 'cols' to determine layout
  if (is.null(layout)) {
    # Make the panel
    # ncol: Number of columns of plots
    # nrow: Number of rows needed, calculated from # of cols
    layout <- matrix(seq(1, cols * ceiling(numPlots/cols)),
                    ncol = cols, nrow = ceiling(numPlots/cols))
  }

 if (numPlots==1) {
    print(plots[[1]])

  } else {
    # Set up the page
    grid.newpage()
    pushViewport(viewport(layout = grid.layout(nrow(layout), ncol(layout))))

    # Make each plot, in the correct location
    for (i in 1:numPlots) {
      # Get the i,j matrix positions of the regions that contain this subplot
      matchidx <- as.data.frame(which(layout == i, arr.ind = TRUE))

      print(plots[[i]], vp = viewport(layout.pos.row = matchidx$row,
                                      layout.pos.col = matchidx$col))
    }
  }
}

r <- get_errors(df, eps)
xi <- get_info(r, 'xi', eps)
xib <- get_info(r, 'xib', eps)
fun <- get_info(r, 'fun', eps)
cha <- get_info(r, 'cha', eps)

gxi <- ggplot(xi, aes(x=eps, y=mean)) + geom_errorbar(aes(ymin=mean-sd, ymax=mean+sd), width=.1) + geom_line() + geom_point() + scale_x_log10() + coord_cartesian(ylim=c(0.14,0.2)) + ggtitle("Bounded")
gxib <- ggplot(xib, aes(x=eps, y=mean)) + geom_errorbar(aes(ymin=mean-sd, ymax=mean+sd), width=.1) + geom_line() + geom_point() + scale_x_log10() + coord_cartesian(ylim=c(0.14,0.2)) + ggtitle("Bounded, Biased")
gfun <- ggplot(fun, aes(x=eps, y=mean)) + geom_errorbar(aes(ymin=mean-sd, ymax=mean+sd), width=.1) + geom_line() + geom_point() + scale_x_log10() + coord_cartesian(ylim=c(0, 1)) + ggtitle("Functional")
gcha <- ggplot(cha, aes(x=eps, y=mean)) + geom_errorbar(aes(ymin=mean-sd, ymax=mean+sd), width=.1) + geom_line() + geom_point() + scale_x_log10() + coord_cartesian(ylim=c(0.14,0.2)) + ggtitle("Ridge")
ggsave('Adult-4-plots.png', multiplot(gxi, gxib, gfun, gcha, cols=2), width=10, height=6, units="in")

#M - how bad is releasing a model that isn't accurate enough?
#N - what is the maximum value of epsilon permissible to release?
exp_mech <- function(M, N, epsilon, epsilon_bag){
	n <- length(epsilon_bag)
	base <- exp(-epsilon/(2*M))
	index_areas <- epsilon_bag*(base^M) + (base^N - base^epsilon_bag)/log(base)
	c_index_areas <- cumsum(index_areas)
	c_index_areas <- c_index_areas / c_index_areas[n]
	u <- runif(1)
	i <- min(n, findInterval(u, c_index_areas)+1)
	e_i <- epsilon_bag[i]
	t1 <- e_i * base^M
	t2 <- (base^N-base^e_i) / log(base)
	if(runif(1, min=0, max=1) <= t1 / (t1+t2)){
		#Oh no!
		return(list(idx=i, eps=runif(1, min=0, max=e_i)))
	}else{
		add <- e_i + rexp(1, -log(base)) %% (N-e_i)
		return(list(idx=i, eps=add))
	}
}

exp_mech_plot <- function(M, N, epsilon, epsilon_bag){
	n <- length(epsilon_bag)
	base <- exp(-epsilon/(2*M))
	index_areas <- epsilon_bag*(base^M) + (base^N - base^epsilon_bag)/log(base)
	index_areas <- index_areas / sum(index_areas)
	return(list(probs=data.frame(epsilon=epsilon, Spent=epsilon_bag, Probability=index_areas), 
		exp=data.frame(epsilon=epsilon, Expected_Value=sum(index_areas * epsilon_bag) + epsilon)))
}
get_expo_graphs <- function(epsilon_bag=c(0.8, 1.6, 0.1), eps2 = 1:50/10, M=4, N=4){
	exp <- data.frame()
	prob <- data.frame()
	for(e in eps2){
		Q <- exp_mech_plot(M,N, e, epsilon_bag)
		exp <- rbind(exp, Q$exp)
		prob <- rbind(prob, Q$probs)
	}
	exp2 <- exp
	exp$Strat <- "E.M."
	exp2$Strat <- "Rand"
	exp2$Expected_Value	<- mean(epsilon_bag)
	exp <- rbind(exp, exp2)
	prob$Spent <- as.character(prob$Spent)
	return(list(p=ggplot(prob, aes(x=epsilon, y=Probability, group=Spent, color=Spent)) + geom_line() + coord_cartesian(ylim=c(0,0.5)) + ggtitle("Probability of Algorithm Selection"),
	e=ggplot(exp, aes(x=epsilon, y=Expected_Value, group=Strat, color=Strat)) + geom_line() + ggtitle("Expected Privacy Usage")))
}
G <- get_expo_graphs(c(0.8, 1.6, 0.1, 40), 1:100/10, M=40, N=40)
G2 <- get_expo_graphs(c(0.8, 1.6, 0.1), 1:100/10, M=40, N=40)
ggsave('Adult-4-exponential.png', multiplot(G2$p, G$p, G2$e, G$e, cols=2), width=10, height=6, units="in")