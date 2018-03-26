
library(nloptr);
library(VGAM);
library(compiler);
library(pROC);
library(RWeka);
library(MASS);
library(pph);
library(stringr);
library(foreign);

# training <- read.arff("~/data/iwpc_train.arff");
# validation <- read.arff("~/data/iwpc_valid.arff");
vkoattr <- c("vkorc1=CT", "vkorc1=TT");
cypattr <- c("cyp2c9=13", "cyp2c9=12", "cyp2c9=23", "cyp2c9=33", "cyp2c9=22");

tabulateUtilityExperiments <- function(dir) {
	
	ttrdata <- matrix(0, nrow=4, ncol=5);
	rownames(ttrdata) <- c("fixed", "lr", "dplr", "pdata");
	colnames(ttrdata) <- c("0.25", "1.00", "5.00", "20.00", "100.00");
	strokedata <- matrix(0, nrow=4, ncol=5);
	rownames(strokedata) <- c("fixed", "lr", "dplr", "pdata");
	colnames(strokedata) <- c("0.25", "1.00", "5.00", "20.00", "100.00");
	bleeddata <- matrix(0, nrow=4, ncol=5);
	rownames(bleeddata) <- c("fixed", "lr", "dplr", "pdata");
	colnames(bleeddata) <- c("0.25", "1.00", "5.00", "20.00", "100.00");
	deathdata <- matrix(0, nrow=4, ncol=5);
	rownames(deathdata) <- c("fixed", "lr", "dplr", "pdata");
	colnames(deathdata) <- c("0.25", "1.00", "5.00", "20.00", "100.00");
	
	rawdatas <- list();
	
	curdir <- getwd();
	setwd(dir);
	filenames <- list.files(pattern=".*-eps.*\\.Robject");
	for(file in filenames) {
		match <- str_match(file, ".*-eps\\.([0-9]+\\.[0-9]+)-.*");
		epsstr <- match[2];
		eps <- as.numeric(match[2]);
		data <- dget(file);
		rawdata <- data$rawdata;

		rawdatas[[epsstr]] <- rawdata;		
				
		for(rname in rownames(ttrdata)) {		
			ttrdata[rname, epsstr] <- mean(rawdata[,sprintf("ttrs.%s", rname)]);
			strokedata[rname, epsstr] <- mean(rawdata[,sprintf("strokes.%s", rname)]);
			bleeddata[rname, epsstr] <- mean(rawdata[,sprintf("ichs.%s", rname)]) + mean(rawdata[,sprintf("echs.%s", rname)]);
			deathdata[rname, epsstr] <- mean(rawdata[,sprintf("deaths.%s", rname)]);
			
			rawdatas[[epsstr]][,sprintf("bleeds.%s", rname)] <- rawdata[,sprintf("ichs.%s", rname)] + rawdata[,sprintf("echs.%s", rname)];
		}
	}

	metrics <- c("ttrs", "strokes", "bleeds", "deaths");
	controls <- c("lr", "dplr", "pdata");
	paramvals <- colnames(ttrdata);
	fixed <- "fixed";
	
	rrplots <- list();
	absplots <- list();
	for(metric in metrics) {
		absplots[sprintf("%s.fixed", metric)] <- "";
		for(eps in paramvals) {
			for(control in controls) {
				plotstr <- sprintf("%s.%s", metric, control);
				rrplots[plotstr] <- "";
				absplots[plotstr] <- "";
			}
			absplots[sprintf("%s.fixed", metric)] <- sprintf("%s\n(%s, %f)", absplots[sprintf("%s.fixed", metric)], eps, mean(rawdatas[[eps]][,sprintf("%s.fixed", metric)]));
		}
	}
	
	pvaldata <- list();
	for(metric in metrics) {
		cat(sprintf("metric: %s\n", metric));
		for(eps in paramvals) {
			cat(sprintf("\teps=%s\n", eps));
			for(control in controls) {
				fixedrisk <- sprintf("%s.fixed", metric);
				contrisk <- sprintf("%s.%s", metric, control);
				rr <- rawdatas[[eps]][,contrisk] / rawdatas[[eps]][,fixedrisk];
				diff <- rawdatas[[eps]][,contrisk] - rawdatas[[eps]][,fixedrisk];
				pval.rr.lt <- t.test(x=rr-1.0, alternative="l", conf.level=0.95)$p.value;
				pval.rr.gt <- t.test(x=rr-1.0, alternative="g", conf.level=0.95)$p.value;
				pval.diff.lt <- t.test(x=diff, alternative="l", conf.level=0.95)$p.value;
				pval.diff.gt <- t.test(x=diff, alternative="g", conf.level=0.95)$p.value;
				cat(sprintf("\t\t%s: var = %f rr = %f (p< = %f, p> = %f), ", control, var(rawdatas[[eps]][,contrisk]), mean(rr), pval.rr.lt, pval.rr.gt));
				cat(sprintf("diff = %f (p< = %f, p> = %f)\n", mean(diff), pval.diff.lt, pval.diff.gt));
				
				pvaldata[[sprintf("%s.%s.%s", metric, eps, control)]] <- sprintf("%s: rr = %f (p< = %f, p> = %f), diff = %f (p< = %f, p> = %f)", control, mean(rr), pval.rr.lt, pval.rr.gt, mean(diff), pval.diff.lt, pval.diff.gt);
				
				plotstr <- sprintf("%s.%s", metric, control);
				rrplots[plotstr] <- sprintf("%s\n(%s, %f)", rrplots[plotstr], eps, mean(rr));
				absplots[plotstr] <- sprintf("%s\n(%s, %f)", absplots[plotstr], eps, mean(rawdatas[[eps]][,contrisk]));
			}
		}
	}
	
	print(absplots);

	filename <- sprintf("tabulation.Robject");
	dput(list(ttrdata, strokedata, bleeddata, deathdata, rrplots, absplots, pvaldata), file=filename);

	setwd(curdir);
}

tabulateReverseLrExperiments <- function(dir) {
	
	results <- list();
	
	curdir <- getwd();
	setwd(dir);
	filenames <- list.files(pattern="runMultipleReverseClassifiers-eps.*\\.Robject");
	for(file in filenames) {
		match <- str_match(file, "runMultipleReverseClassifiers-eps\\.([0-9]+\\.[0-9]+)-missing\\.([0-9]+)-guess\\.(.+)-nruns\\.([0-9]+)-statsfun\\.(.+)-(.+)\\.Robject");

		eps <- match[2];
		missing <- match[3];
		guess <- match[4];
		statsfun <- match[6];

		if(is.null(results[[guess]])) results[[guess]] <- list();
		if(is.null(results[[guess]][[statsfun]]))results[[guess]][[statsfun]] <- list();
		if(is.null(results[[guess]][[statsfun]][[missing]])) results[[guess]][[statsfun]][[missing]] <- list();
		if(is.null(results[[guess]][[statsfun]][[missing]][[eps]])) results[[guess]][[statsfun]][[missing]][[eps]] <- list();
	}

	for(file in filenames) {
		match <- str_match(file, "runMultipleReverseClassifiers-eps\\.([0-9]+\\.[0-9]+)-missing\\.([0-9]+)-guess\\.(.+)-nruns\\.([0-9]+)-statsfun\\.(.+)-(.+)\\.Robject");

		eps <- match[2];
		missing <- match[3];
		guess <- match[4];
		statsfun <- match[6];
		rawdata <- dget(file);
						
		bacc <- as.numeric(str_match(rawdata$baseline,".*, bdiffacc: [-]?[0-9]+\\.[0-9]+ \\(([0-9]+\\.[0-9]+) - .*")[2]);
		bauc <- as.numeric(str_match(rawdata$baseline,".*, bdiffauc: [-]?[0-9]+\\.[0-9]+ \\(([0-9]+\\.[0-9]+) - .*")[2]);

		results[[guess]][[statsfun]][[missing]][[eps]]$desc <- sprintf("eps=%s, missing=%s, guess=%s, statsfun=%s", eps, missing, guess, statsfun);
		results[[guess]][[statsfun]][[missing]][[eps]]$summary <- sprintf("\t%s\n", rawdata$summary);
		results[[guess]][[statsfun]][[missing]][[eps]]$baseline <- sprintf("\t%s\n\n", rawdata$baseline);
		results[[guess]][[statsfun]][[missing]][[eps]]$accdiff <- sprintf("\ttest against non-private acc: %f", t.test.silent(x=rawdata$accs-bacc, alternative="l", conf.level=0.95));
		results[[guess]][[statsfun]][[missing]][[eps]]$aucdiff <- sprintf("\ttest against non-private auc: %f", t.test.silent(x=rawdata$auc-bauc, alternative="l", conf.level=0.95));
	}

	for(guess in results) {
		for(statsfun in guess) {
			for(missing in statsfun) {
				cat("---------------------------------\n");
				for(eps in missing) {		
					cat(sprintf("%s:\n", eps$desc));
					cat(sprintf("\t%s", eps$summary));
					cat(sprintf("\t%s\n\t%s\n", eps$accdiff, eps$aucdiff));
					cat(sprintf("\t%s", eps$baseline));					
				}
			}
		}
	}

	setwd(curdir);
}

tabulatePdataExperiments <- function(dir) {
	
	results <- list();
	
	curdir <- getwd();
	setwd(dir);
	filenames <- list.files(pattern="runAllAnonDatasetComparisons.*\\.Robject");
	for(file in filenames) {
		match <- str_match(file, "runAllAnonDatasetComparisons-.*-eps\\.([0-9]+\\.[0-9]+)-.*missing\\.([0-9]+)-guess\\.(.+).*-evalfun\\.(.+)-.*");

		eps <- match[2];
		missing <- match[3];
		guess <- match[4];
		statsfun <- match[5];

		if(is.null(results[[guess]])) results[[guess]] <- list();
		if(is.null(results[[guess]][[statsfun]]))results[[guess]][[statsfun]] <- list();
		if(is.null(results[[guess]][[statsfun]][[missing]])) results[[guess]][[statsfun]][[missing]] <- list();
		if(is.null(results[[guess]][[statsfun]][[missing]][[eps]])) results[[guess]][[statsfun]][[missing]][[eps]] <- list();
	}

	for(file in filenames) {
		match <- str_match(file, "runAllAnonDatasetComparisons-.*-eps\\.([0-9]+\\.[0-9]+)-.*missing\\.([0-9]+)-guess\\.(.+).*-evalfun\\.(.+)-.*");

		eps <- match[2];
		missing <- match[3];
		guess <- match[4];
		statsfun <- match[5];
		rawdata <- dget(file);
						
		bacc <- as.numeric(str_match(rawdata$baseline,".*, bdiffacc: [-]?[0-9]+\\.[0-9]+ \\(([0-9]+\\.[0-9]+) - .*")[2]);
		bauc <- as.numeric(str_match(rawdata$baseline,".*, bdiffauc: [-]?[0-9]+\\.[0-9]+ \\(([0-9]+\\.[0-9]+) - .*")[2]);

		results[[guess]][[statsfun]][[missing]][[eps]]$desc <- sprintf("eps=%s, missing=%s, guess=%s, statsfun=%s", eps, missing, guess, statsfun);
		results[[guess]][[statsfun]][[missing]][[eps]]$summary <- sprintf("\t%s\n", rawdata$summary);
		results[[guess]][[statsfun]][[missing]][[eps]]$baseline <- sprintf("\t%s\n\n", rawdata$baseline);
		results[[guess]][[statsfun]][[missing]][[eps]]$accdiff <- sprintf("\ttest against non-private acc: %f", t.test.silent(x=rawdata$accs-bacc, alternative="l", conf.level=0.95));
		results[[guess]][[statsfun]][[missing]][[eps]]$aucdiff <- sprintf("\ttest against non-private auc: %f", t.test.silent(x=rawdata$auc-bauc, alternative="l", conf.level=0.95));
	}

	for(guess in results) {
		for(statsfun in guess) {
			for(missing in statsfun) {
				cat("---------------------------------\n");
				for(eps in missing) {		
					cat(sprintf("%s:\n", eps$desc));
					cat(sprintf("\t%s", eps$summary));
					cat(sprintf("\t%s\n\t%s\n", eps$accdiff, eps$aucdiff));
					cat(sprintf("\t%s", eps$baseline));					
				}
			}
		}
	}

	setwd(curdir);
}

fixupPrivateDatasetNames <- function() {
	
	filenames <- list.files(pattern="trainep25.2-.*\\.csv");
	for(file in filenames) {
		match <- str_match(file, "trainep25.2-([0-9]+).csv");
		file.rename(file, sprintf("trainep0.25.2-%s.csv", match[2]));
	}
	
	filenames <- list.files(pattern="trainep([0-9]+).2-([0-9]+).csv");
	for(file in filenames) {
		match <- str_match(file, "trainep([0-9]+).2-([0-9]+).csv");
		file.rename(file, sprintf("trainep%.2f.2-%s.csv", as.numeric(match[2]), match[3]));
	}
}

cyp2wild <- function(data) {
	
	# should be in the right order
	cypattrs <- c("cyp2c9=13", "cyp2c9=12", "cyp2c9=23", "cyp2c9=33", "cyp2c9=22");
	
	dnames <- dimnames(data)[[2]];
	cyps <- apply(as.array(1:nrow(data)), 1, function(x) 1 - sum(data[x,cypattrs]));
	
	cypidx <- which(dnames == cypattrs[1]);
	newnames <- dnames[!(dnames %in% cypattrs)];
	newnames <- append(newnames, "cyp2c9=wild", after=cypidx-1);
	print(newnames);
	
	data$`cyp2c9=wild` <- cyps;
	data <- data[,newnames];
	
	return(data);
}

remove_outliers <- function(x, na.rm = TRUE, ...) {
  qnt <- quantile(x, probs=c(.25, .75), na.rm = na.rm, ...)
  H <- 1.5 * IQR(x, na.rm = na.rm)
  y <- x
  y[x < (qnt[1] - H)] <- NA
  y[x > (qnt[2] + H)] <- NA
  na.omit(y)
}

cyp2c9_nom <- function(s) { 
	
	attrs <- c("cyp2c9=13", "cyp2c9=12", "cyp2c9=23", "cyp2c9=33", "cyp2c9=22");
	
	if(sum(s[attrs]) <= 0) { 
		return("cyp2c9=11");
	} else { 
		return(attrs[which.max(s[attrs])]); 
	}
}

vkorc1_nom <- function(s) { 
	
	attrs <- c("vkorc1=CT", "vkorc1=TT");
	
	if(sum(s[attrs]) <= 0) { 
		return("vkorc1=CC");
	} else { 
		return(attrs[which.max(s[attrs])]); 
	}
}

bias <- function(clfun, data, t) {
	
	data <- as.matrix(data);
	doses <- apply(as.array(1:nrow(data)), 1, function(x) WarfDose(clfun(data[x,])));
	
	return(list(pct=sum(doses > t)/nrow(data), mean=mean(doses)));
}

fixupNominalDataset <- function(data) {
	
	dnames <- apply(as.array(dimnames(data)[[2]]), 1, function(x) sub("\\.", "=", x));
	dnames <- append(dnames, c("race=black", "race=asian", "cyp2c9=13", "cyp2c9=12", "cyp2c9=23", "cyp2c9=33", "cyp2c9=22", "vkorc1=CT", "vkorc1=TT"));
	
	racelevels <- levels(data[,"race"]);
	raceblack <- match("race=black", racelevels);
	raceasian <- match("race=asian", racelevels);	
	
	cyplevels <- levels(data[,"cyp2c9"]);
	cyp13 <- match("cyp2c9=13", cyplevels);
	cyp12 <- match("cyp2c9=12", cyplevels);
	cyp23 <- match("cyp2c9=23", cyplevels);
	cyp33 <- match("cyp2c9=33", cyplevels);	
	cyp22 <- match("cyp2c9=22", cyplevels);	

	vkolevels <- levels(data[,"vkorc1"]);
	vkoCT <- match("vkorc1=CT", vkolevels);
	vkoTT <- match("vkorc1=TT", vkolevels);
	
	data$`race=black` <- apply(as.array(data[,"race"]), 1, function(x) ifelse(x == raceblack, 1, 0));
	data$`race=asian` <- apply(as.array(data[,"race"]), 1, function(x) ifelse(x == raceasian, 1, 0));	
	data$`cyp2c9=13` <- apply(as.array(data[,"cyp2c9"]), 1, function(x) ifelse(x == cyp13, 1, 0));
	data$`cyp2c9=12` <- apply(as.array(data[,"cyp2c9"]), 1, function(x) ifelse(x == cyp12, 1, 0));
	data$`cyp2c9=23` <- apply(as.array(data[,"cyp2c9"]), 1, function(x) ifelse(x == cyp23, 1, 0));
	data$`cyp2c9=33` <- apply(as.array(data[,"cyp2c9"]), 1, function(x) ifelse(x == cyp33, 1, 0));
	data$`cyp2c9=22` <- apply(as.array(data[,"cyp2c9"]), 1, function(x) ifelse(x == cyp22, 1, 0));
	data$`vkorc1=CT` <- apply(as.array(data[,"vkorc1"]), 1, function(x) ifelse(x == vkoCT, 1, 0));
	data$`vkorc1=TT` <- apply(as.array(data[,"vkorc1"]), 1, function(x) ifelse(x == vkoTT, 1, 0));	
	
	data <- data[,!(dnames %in% c("vkorc1", "cyp2c9", "race"))];	
	fnames <- dimnames(data)[[2]];
	
	data <- data[,append(c("race=black", "race=asian"), append(fnames[!(fnames %in% c("dose", "decr", "race=black", "race=asian"))], c("decr", "dose")))];
	
	for(name in fnames) {
		if(is.factor(data[,name])) {
			data[,name] <- as.numeric(as.character(data[,name]));
		}
	}
	
	return(data);
}

discretizeDataset <- function(data, ncut) {

	onames <- dimnames(data)[[2]];
	newdata <- list();
	
	for(name in onames) {
		
		uniquecol <- unique(data[,name]);
		nunique <- length(uniquecol);
		
		if(nunique <= ncut) {
			
			newdata[[name]] <- factor(data[,name]);
		} else {
			
			stepsize <- (max(data[,name])-min(data[,name]))/ncut;
			labs <- apply(as.array(1:ncut), 1, function(i) as.character((min(data[,name])+(i-1)*stepsize+stepsize/2)));
			newdata[[name]] <- cut(data[,name], breaks=ncut, labels=labs);
		}
	}
	
	newdata <- data.frame(newdata);
	
	names(newdata) <- onames;
	
	return(newdata);
}

numericFromNominal <- function(data) {
	
	onames <- dimnames(data)[[2]];
	newdata <- list();
	
	n <- nrow(data);
	
	for(name in onames) {

		if(is.factor(data[,name])) {			
			newdata[[name]] <- as.numeric(as.character(data[,name]));
		} else {
			newdata[[name]] <- data[,name];
		}		
	}
	
	newdata <- data.frame(newdata);
	
	names(newdata) <- onames;
	
	return(newdata);
	
}

cyp2c9levs <- c("cyp2c9=13", "cyp2c9=12", "cyp2c9=23", "cyp2c9=33", "cyp2c9=22", "cyp2c9=11");
vkorc1levs <- c("vkorc1=CT", "vkorc1=TT", "vkorc1=CC");
racelevs <- c("race=black", "race=asian", "race=white");

cyp2c9cat <- function(s) {
	
	lev <- which(s[c("cyp2c9=13", "cyp2c9=12", "cyp2c9=23", "cyp2c9=33", "cyp2c9=22")] > 0);
	if(length(lev) <= 0) {
		return(factor("cyp2c9=11", levels=cyp2c9levs));
	} else {
		return(factor(cyp2c9levs[lev], levels=cyp2c9levs));
	}
}

vkorc1cat <- function(s) {
	
	lev <- which(s[c("vkorc1=CT", "vkorc1=TT")] > 0);
	if(length(lev) <= 0) {
		return(factor("vkorc1=CC", levels=vkorc1levs));
	} else {
		return(factor(vkorc1levs[lev], levels=vkorc1levs));
	}
}

racecat <- function(s) {
	
	lev <- which(s[c("race=black", "race=asian")] > 0);
	if(length(lev) <= 0) {
		return(factor("race=white", levels=racelevs));
	} else {
		return(factor(racelevs[lev[sample.int(length(lev), 1)]], levels=racelevs));
	}
}

nominalGenes <- function(data) {
		
	cyp2c9s <- apply(as.array(1:nrow(data)), 1, function(x) cyp2c9cat(data[x,]));	
	vkorc1s <- apply(as.array(1:nrow(data)), 1, function(x) vkorc1cat(data[x,]));
	races <- apply(as.array(1:nrow(data)), 1, function(x) racecat(data[x,]));
	
	onames <- dimnames(data)[[2]];
	data <- data.frame(data);
	names(data) <- onames;		
	
	data$race <- races;
	data$cyp2c9 <- cyp2c9s;
	data$vkorc1 <- vkorc1s;	
	
	data <- data[,!(dimnames(data)[[2]] %in% c("race=black", "race=asian", "cyp2c9=13", "cyp2c9=12", "cyp2c9=23", "cyp2c9=33", "cyp2c9=22", "vkorc1=CT", "vkorc1=TT"))];
	data <- data[,append(c(ncol(data)-2), append(1:(ncol(data)-4), append(c(ncol(data)-1), append(c(ncol(data)), c(ncol(data)-3)))))]
	
	data$age <- as.factor(data$age);
	data$amiodarone <- as.factor(data$amiodarone);
	data$decr <- as.factor(data$decr);

	return(data);
}

diffpDatasetOld <- function(data, class, ep, ncut) {
	
	#data <- data[,!(dimnames(data)[[2]] %in% c("amiodarone", "decr"))];
		
	cyp2c9s <- apply(as.array(1:nrow(data)), 1, function(x) cyp2c9cat(data[x,]));	
	vkorc1s <- apply(as.array(1:nrow(data)), 1, function(x) vkorc1cat(data[x,]));
	
	onames <- dimnames(data)[[2]];
	data <- data.frame(data);
	names(data) <- onames;
	
	data <- nominalGenes(data);
	
	#data$cyp2c9 <- cyp2c9s;
	#data$vkorc1 <- vkorc1s;	
	
	#data <- data[,!(dimnames(data)[[2]] %in% c("cyp2c9=13", "cyp2c9=12", "cyp2c9=23", "cyp2c9=33", "cyp2c9=22", "vkorc1=CT", "vkorc1=TT"))];
	#data <- data[,append(1:(ncol(data)-3), append(c(ncol(data)-1), append(c(ncol(data)), c(ncol(data)-2))))]	
	
	t <- lapply(dimnames(data)[[2]], 
		function(x) { 
			if(!is.factor(data[,x])) {
				if(length(unique(data[,x])) > 10) {
					stepsize <- (max(data[,x])-min(data[,x]))/ncut;
					labs <- apply(as.array(1:ncut), 1, function(i) as.character((min(data[,x])+(i-1)*stepsize+stepsize/2)));
					return(cut(data[,x], breaks=ncut, labels=labs));
				} else {
					return(as.factor(data[,x]));
				}				
			} else {
				return(data[,x]);
			}
		});
	
	t <- data.frame(t);
	names(t) <- dimnames(data)[[2]];
	t <- na.omit(t);	
	
	dpt <- pdata(t, target=which(names(t) == class), eps=ep, k=ncol(t)-1, verbose=TRUE);
	names(dpt) <- dimnames(t)[[2]];
	
	return(dpt);
}

diffpDataset <- function(data, ep, ncut) {
	
	#data <- data[,!(dimnames(data)[[2]] %in% c("amiodarone", "decr"))];

	onames <- dimnames(data)[[2]];
	
	data <- discretizeDataset(data, 2);
	names(data) <- dimnames(data)[[2]];
	data <- na.omit(data);	
	
	data <- numericFromNominal(pdata(data, target=1, eps=ep, k=ncol(data)-1, verbose=TRUE));
	names(data) <- onames;
	data <- as.matrix(data);
	#data <- fixupVectorAttributes(data, getVectorAttributes(onames), 1);
	
	return(data);
}

numericDiffpDataset <- function(data, class, ep) {
	
	dnames <- dimnames(data)[[2]];
	data <- data.frame(data);
	names(data) <- dnames;
	#data <- nominalGenes(data);
	
	#data$`cyp2c9=13` <- as.factor(data$`cyp2c9=13`);
	#data$`race=black` <- as.factor(data$`race=black`);
	#data$`race=asian` <- as.factor(data$`race=asian`);
	data$decr <- as.factor(data$decr);
	#data$amiodarone <- as.factor(data$amiodarone);
	#data$decr <- as.factor(data$decr);	
	
	factors <- apply(as.array(1:ncol(data)), 1, function(i) { ifelse(!is.factor(data[,i]), max(data[,i]), NA) });
	
	for(i in 1:ncol(data)) {
		
		if(is.factor(data[,i])) next;
		
		data[,i] <- data[,i] / factors[i];
	}			
	
	dpt <- pdata(data, target=which(names(data) == class), eps=ep, k=ncol(data), verbose=TRUE);	
	names(dpt) <- dnames;
	dpt$decr <- as.numeric(as.character(dpt$decr));	
	
	data <- dpt;
	
	for(i in 1:ncol(data)) {
		
		if(is.na(factors[i])) next;
		
		data[,i] <- data[,i] * factors[i];
	}
	
	return(data);
	#return(fixupNominalDataset(data));
}

generateDiffpDatasets <- function(data, class, ep, ncut, nsets, fileprefix) {
	
	for(i in 1:nsets) {
		
		set.seed(i);
		
		filename <- paste0(fileprefix, "-", as.character(i), ".csv");
		
		print(sprintf("working on i = %d", i));
		
		dpd <- diffpDataset(data, class, ep, ncut);
		print(summary(dpd));
						
		fixed <- fixupNominalDataset(dpd);
		print(sprintf("writing to %s", filename));
		print("------------------------");
		
		write.csv(fixed, filename, row.names=FALSE);		
	}
}

evaluateDiffpDatasets <- function(fileprefix, nsets, validation, class) {
	
	accs <- numeric(0);
	
	for(i in 1:nsets) {
		
		filename <- paste0(fileprefix, "-", as.character(i), ".csv");
		
		print(sprintf("loading %s", filename));
		
		data <- read.csv(filename);
		names(data) <- names(validation);
		clfun <- diffpLR(data, class, 0);
		
		stats <- regstats(function(x) WarfDose(clfun(x)), validation, class);
		accs <- append(accs, stats$relative);
		
		print(sprintf("current mean acc: %f", mean(accs)));
	}
	
	return(accs);
}

fmeasure <- function(confusion, data, guess) {
	
	measures <- numeric(0);
	
	for(i in 1:nrow(confusion)) {
		
		totalpos <- sum(as.numeric(data[,guess]) == i);
		totalneg <- nrow(data) - totalpos;
				
		tp <- confusion[i,i] / totalpos;
		fp <- (sum(confusion[,i]) - confusion[i,i]) / totalneg;		
		
		fn <- 0;
		for(j in 1:nrow(confusion)) {
			if(i == j) next;
			
			fn <- fn + confusion[j, i];
		}
		fn <- fn / totalpos;
		
		prec <- tp / (tp + fp);
		recall <- tp / (tp + fn);
		
		fm <- 2*prec*recall / (prec + recall);
		if(is.nan(fm)) fm <- 0;
		
		measures <- append(measures, fm);
	}
	
	return(mean(measures));
};

fmeasurevector <- function(confusion, data, guess) {
	
	measures <- numeric(0);
	n <- sum(confusion);
	
	for(i in 1:nrow(confusion)) {
		
		if(i == 1) {
			totalpos <- sum(apply(as.matrix(data[,guess]), 1, function(x) ifelse(sum(as.numeric(x)) == 0, 1, 0) ));
			totalneg <- n - totalpos;
		} else {
			totalpos <- sum(as.numeric(data[,guess[i-1]]) > 0);
			totalneg <- n - totalpos;
		}
				
		tp <- confusion[i,i] / sum(confusion[,i]);
		fp <- (sum(confusion[,i]) - confusion[i,i]) / sum(confusion[,i]);		
		
		fn <- 0;
		for(j in 1:nrow(confusion)) {
			if(i == j) next;
			
			fn <- fn + confusion[j, i];
		}
		fn <- fn / totalpos;
		fn <- (totalpos - confusion[i,i]) / totalpos;
		
		prec <- tp / (tp + fp);
		recall <- tp / (tp + fn);
		
		fm <- 2*prec*recall / (prec + recall);
		if(is.nan(fm)) fm <- 0;
		
		measures <- append(measures, fm);
	}
	
	return(mean(measures));
};

aucroc <- function(confusion, data, guess) {
	
	measures <- numeric(0);
	
	for(i in 1:nrow(confusion)) {
		
		totalpos <- sum(as.numeric(data[,guess]) == i);
		totalneg <- nrow(data) - totalpos;
				
		tp <- confusion[i,i] / totalpos;
		fp <- (sum(confusion[,i]) - confusion[i,i]) / totalneg;
		
		#print(sprintf("tp: %f, fp: %f, totalpos: %d, totalneg: %d, sum: %d", tp, fp, totalpos, totalneg, sum(confusion[,i])));

		auc <- 1/2*tp*fp + 1/2*(tp+1)*(1-fp);
		#if(auc < 0.5) auc <- 1 - auc;
		
		if(is.nan(auc)) fm <- 0;
		
		measures <- append(measures, auc);
	}
	
	return(mean(measures));
};

aucrocvector <- function(confusion, data, guess) {
	
	measures <- numeric(0);
	n <- sum(confusion);
	
	for(i in 1:nrow(confusion)) {
		
		if(i == 1) {
			totalpos <- sum(apply(data[,guess], 1, function(x) 1 - sum(x)));
			totalneg <- n - totalpos;
		} else {
			totalpos <- sum(as.numeric(data[,guess[i-1]]) > 0);
			totalneg <- n - totalpos;
		}
				
		tp <- confusion[i,i] / totalpos;
		fp <- (sum(confusion[,i]) - confusion[i,i]) / sum(confusion[,i]);
		
		#print(sprintf("tp: %f, fp: %f, totalpos: %d, totalneg: %d, sum: %d", tp, fp, totalpos, totalneg, sum(confusion[,i])));

		auc <- 1/2*tp*fp + 1/2*(tp+1)*(1-fp);
		#if(auc < 0.5) auc <- 1 - auc;
		
		if(is.nan(auc)) auc <- 0;
		
		measures <- append(measures, auc);
	}
	
	return(mean(measures));
};

genfolds <- function(data, nfolds) {
	
	folds <- list();
	
	n <- as.integer(nrow(data)/nfolds);
	
	data <- data[sample(nrow(data)),];
	
	for(i in 1:nfolds) {
		
		test <- data[((i-1)*n):(i*n),];
		train <- data[!(1:(nfolds*n) %in% ((i-1)*n):(i*n)),];
		
		folds[[i]] <- list(test = test, train = train);
	}
	
	folds;
}

doKfoldvalidation <- function(data, class, nfolds, eps, nsamples) {
	
	folds <- genfolds(data, nfolds);
	
	sqerr <- numeric(0);
	absacc <- numeric(0);
	relacc <- numeric(0);
	
	for(i in 1:nfolds) {
		
		for(j in 1:nsamples) {
			clfun <- diffpLR(folds[[i]]$test, class, eps);
			
			stats <- regstats(clfun, folds[[i]]$train, class);
			
			sqerr <- append(sqerr, stats$square);
			absacc <- append(absacc, stats$absolute);
			relacc <- append(relacc, stats$relative);
		}
		
		print(rbind(list(square = mean(sqerr), absolute = mean(absacc), relative = mean(relacc))));
	}
	
	return(list(rmse = sqrt(mean(sqerr)), absolute = mean(absacc), relative = mean(relacc)));
}

runAllAnonDatasetRegress <- function(fileprefix, nsets, validation, class) {

	accs <- numeric(0);
	aucs <- numeric(0);
	fscs <- numeric(0);	
	
	validation <- as.matrix(validation);
	
	for(i in 1:nsets) {
		
		filename <- paste0(fileprefix, "-", as.character(i), ".csv");
		
		#print(sprintf("loading %s", filename));
		
		data <- read.csv(filename);
		names(data) <- dimnames(validation)[[2]];
		data <- as.matrix(data);
		print(summary(data));
		
		clfun <- diffpLR(data, class, 0);
		results <- regstats(clfun, validation, class);
		
		accs <- append(accs, results$relative);
		
		cat(sprintf("%f ", mean(accs)));
	}
	cat("\n");
	
	return(list(accuracy = mean(accs), fmeas = mean(fscs), auc = mean(aucs)));
}

regstats <- function(clfun, data, target) {
	
	sqerrs <- numeric(0);
	abserrs <- numeric(0);
	relerrs <- numeric(0);
	
	data <- as.matrix(data);
	
	for(i in 1:nrow(data)) {
		
		diff <- data[i,target] - clfun(data[i,]);
		
		sqerrs[i] <- diff^2;
		abserrs[i] <- abs(diff);
		relerrs[i] <- abs(diff / data[i,target]);		
	}
	
	return(list(square = mean(sqerrs), absolute = mean(abserrs), relative = mean(relerrs), cirelative = qt(0.975, df=length(relerrs)-1)*sd(relerrs)/sqrt(length(relerrs))));
}

rstatsrev <- function(clfun, data, guess) {
	
	abserrs <- numeric(0);
	nguesstypes <- length(guess);
	errortypes <- matrix(0, nrow = nguesstypes+1, ncol = nguesstypes+1);
	
	data <- as.matrix(data);

	truth <- numeric(0);
	predicted <- numeric(0);
	
	for(i in 1:nrow(data)) {
		
		rguess <- clfun(data[i,]);
		real <- getVectorAttrVal(data[i,], guess);
		abserrs[i] <- ifelse(real == rguess, 1, 0);
		errortypes[real+1, rguess+1] <- errortypes[real+1, rguess+1] + 1;
		truth <- append(truth, real);
		predicted <- append(predicted, rguess);
		
		#if(i %% 100 == 0) cat("(", as.numeric(multiclass.roc(response=truth, predictor=predicted)$auc), mean(abserrs), "), ");
	}
	
	return(list(acc=abserrs, fmeas=fmeasurevector(errortypes, data, guess), auc=as.numeric(multiclass.roc(response=truth, predictor=predicted)$auc), confusion=errortypes));
}

rstatsrev.wildtype <- function(clfun, data, guess) {
	
	abserrs <- numeric(0);
	nguesstypes <- length(guess);
	errortypes <- matrix(0, nrow = 2, ncol = 2);
	
	data <- as.matrix(data);

	truth <- numeric(0);
	predicted <- numeric(0);
	
	for(i in 1:nrow(data)) {
		
		rguess <- ifelse(clfun(data[i,])==0,1,0);
		real <- ifelse(getVectorAttrVal(data[i,], guess)==0,1,0);
		abserrs[i] <- ifelse(real == rguess, 1, 0);
		errortypes[real+1, rguess+1] <- errortypes[real+1, rguess+1] + 1;
		truth <- append(truth, real);
		predicted <- append(predicted, rguess);
		
		#if(i %% 100 == 0) cat("(", as.numeric(multiclass.roc(response=truth, predictor=predicted)$auc), mean(abserrs), "), ");
	}
	
	return(list(acc=mean(abserrs), fmeas=fmeasurevector(errortypes, data, guess), auc=as.numeric(multiclass.roc(response=truth, predictor=predicted)$auc), confusion=errortypes));
}


laprnd <- function(n, mu, b) {
	
	u <- runif(n, 0, 1) - 0.5;
	y <- mu - b * sign(u) * log(1 - 2*abs(u));
	
	return(y);
}

normalize <- function(data, class) {
	
	data <- as.matrix(data);
	
	mins <- apply(as.array(dimnames(data)[[2]]), 1, function(x) min(data[,x]));
	maxs <- apply(as.array(dimnames(data)[[2]]), 1, function(x) max(data[,x]));
	diff <- maxs - mins;
	
	data <- sweep(data, 2, mins);
	data <- sweep(data, 2, diff, '/');
	data <- 2*(data - 0.5);
	
	maxnorm <- max(apply(data, 1, function(x) sqrt(sum(x[1:(length(x)-1)])^2)));
	normvect <- append(numeric(length(dimnames(data)[[2]])-1)+1, 1);
	data <- sweep(data, 2, normvect, '/');
	
	classidx <- match(class, dimnames(data)[[2]]);
	
	return(list(data = data, untransform = function(x) zapsmall(((x*normvect)/2+0.5)*diff+mins, digits=10), dtransform = function(x) (2*((x-mins)/diff-0.5))/normvect, runtransform = function(x) zapsmall(((x*normvect[classidx])/2+0.5)*diff[classidx]+mins[classidx]), digits=10));	
}

rawLR <- function(data, class, epsilon) {
	
	data <- as.matrix(data);
	
	norm <- normalize(data, class);
	data <- norm$data;
	
	trainx <- data[,!(dimnames(data)[[2]] %in% class)];
	trainx <- cbind(as.matrix(trainx), numeric(nrow(trainx))+1);
	
	trainy <- data[,class];
	
	d <- length(dimnames(data)[[2]]);
	
	r0 <- t(trainx) %*% trainx;
	r1 <- -2 * (t(trainx) %*% trainy);
	
	sensitivity <- 2*d*d + 4*d;
	
	noisematrix1 <- NA;
	noisematrix2 <- NA;
	if(epsilon <= 0) {
		noisematrix1 <- matrix(0, nrow = d, ncol = d);
		noisematrix2 <- numeric(d);
	} else {
		noisematrix1 <- matrix(laprnd(d*d, 0, sensitivity * 1/epsilon), nrow = d, ncol = d);
		noisematrix2 <- laprnd(d, 0, sensitivity * 1/epsilon);
	}
	
	coe2 <- r0 + noisematrix1;
	coe2 <- 0.5*(t(coe2) + coe2);
	if(epsilon > 0) {
		coe2 <- coe2 + 4 * sqrt(2) * sensitivity * (1/epsilon) * diag(d);
	} else {
		#coe2 <- coe2 + 10 * sqrt(2) * sensitivity * (1/20) * diag(d);
	}
	coe1 <- r1 + noisematrix2;
			
	eigs <- eigen(coe2);
	vec <- t(apply(eigs$vectors, 1, rev));
	val <- rev(eigs$values);	
		
	del <- val >= 1e-8;
	val <- diag(val);
	val <- val[del,];
	val <- val[,del];
		
	vec <- vec[,del];		
	
	coe2 <- val;
	coe1 <- t(vec) %*% coe1;
	
	noiseobj <- function(w) {
		return(t(w) %*% coe2 %*% w + t(coe1) %*% w);
	}
	
	g0 <- runif(d-sum(!del), 0, 1);
		
	optrsol <- optim(g0, noiseobj, method = "BFGS", control = list(maxit=100000));
	
	g <- optrsol$par;
	
	bestw <- vec %*% g;
	
	w <- bestw[1:length(bestw)-1];
	b <- bestw[length(bestw)];
	
	return(list(w=w, b=b, data=data, dtransform=norm$dtransform, untransform=norm$untransform));
}

rawLRNoNormalize <- function(data, class, epsilon) {
	
	data <- as.matrix(data);
		
	trainx <- data[,!(dimnames(data)[[2]] %in% class)];
	trainx <- cbind(as.matrix(trainx), numeric(nrow(trainx))+1);
	
	trainy <- data[,class];
	
	d <- length(dimnames(data)[[2]]);
	
	r0 <- t(trainx) %*% trainx;
	r1 <- -2 * (t(trainx) %*% trainy);
	
	sensitivity <- 2*d*d + 4*d;
	
	noisematrix1 <- NA;
	noisematrix2 <- NA;
	if(epsilon <= 0) {
		noisematrix1 <- matrix(0, nrow = d, ncol = d);
		noisematrix2 <- numeric(d);
	} else {
		noisematrix1 <- matrix(laprnd(d*d, 0, sensitivity * 1/epsilon), nrow = d, ncol = d);
		noisematrix2 <- laprnd(d, 0, sensitivity * 1/epsilon);
	}
	
	coe2 <- r0 + noisematrix1;
	coe2 <- 0.5*(t(coe2) + coe2);
	if(epsilon > 0) {
		coe2 <- coe2 + 4 * sqrt(2) * sensitivity * (1/epsilon) * diag(d);
	} else {
		#coe2 <- coe2 + 10 * sqrt(2) * sensitivity * (1/20) * diag(d);
	}
	coe1 <- r1 + noisematrix2;
			
	eigs <- eigen(coe2);
	vec <- t(apply(eigs$vectors, 1, rev));
	val <- rev(eigs$values);	
		
	del <- val >= 1e-8;
	val <- diag(val);
	val <- val[del,];
	val <- val[,del];
		
	vec <- vec[,del];		
	
	coe2 <- val;
	coe1 <- t(vec) %*% coe1;
	
	noiseobj <- function(w) {
		return(t(w) %*% coe2 %*% w + t(coe1) %*% w);
	}
	
	g0 <- runif(d-sum(!del), 0, 1);
		
	optrsol <- optim(g0, noiseobj, method = "BFGS", control = list(maxit=100000));
	
	g <- optrsol$par;
	
	bestw <- vec %*% g;
	
	w <- bestw[1:length(bestw)-1];
	b <- bestw[length(bestw)];
	
	return(list(w=w, b=b));
}

diffpLR <- function(data, class, epsilon) {
	
	data <- as.matrix(data);
	
	norm <- normalize(data, class);
	data <- norm$data;
	
	trainx <- data[,!(dimnames(data)[[2]] %in% class)];
	trainx <- cbind(as.matrix(trainx), numeric(nrow(trainx))+1);
	
	trainy <- data[,class];
	
	d <- length(dimnames(data)[[2]]);
	
	r0 <- t(trainx) %*% trainx;
	r1 <- -2 * (t(trainx) %*% trainy);
	
	sensitivity <- 2*d*d + 4*d;
	
	noisematrix1 <- NA;
	noisematrix2 <- NA;
	if(epsilon <= 0) {
		noisematrix1 <- matrix(0, nrow = d, ncol = d);
		noisematrix2 <- numeric(d);
	} else {
		noisematrix1 <- matrix(laprnd(d*d, 0, sensitivity * 1/epsilon), nrow = d, ncol = d);
		noisematrix2 <- laprnd(d, 0, sensitivity * 1/epsilon);
	}
	
	coe2 <- r0 + noisematrix1;
	coe2 <- 0.5*(t(coe2) + coe2);
	if(epsilon > 0) {
		coe2 <- coe2 + 4 * sqrt(2) * sensitivity * (1/epsilon) * diag(d);
	} else {
		#coe2 <- coe2 + 10 * sqrt(2) * sensitivity * (1/20) * diag(d);
	}
	coe1 <- r1 + noisematrix2;
			
	eigs <- eigen(coe2);
	vec <- t(apply(eigs$vectors, 1, rev));
	val <- rev(eigs$values);	
		
	del <- val >= 1e-8;
	val <- diag(val);
	val <- val[del,];
	val <- val[,del];
		
	vec <- vec[,del];		
	
	coe2 <- val;
	coe1 <- t(vec) %*% coe1;
	
	noiseobj <- function(w) {
		return(t(w) %*% coe2 %*% w + t(coe1) %*% w);
	}
	
	g0 <- runif(d-sum(!del), 0, 1);
		
	optrsol <- optim(g0, noiseobj, method = "BFGS", control = list(maxit=100000));
	
	g <- optrsol$par;
	
	bestw <- vec %*% g;
	
	w <- bestw[1:length(bestw)-1];
	b <- bestw[length(bestw)];
	
	clfun <- cmpfun(function(x) {
		
		x <- norm$dtransform(as.numeric(x));
		
		if(length(x) == d) {
			x <- x[!(dimnames(data)[[2]] %in% class)];
		}
		
		return(norm$runtransform(b + x %*% w));
	});
	
	return(clfun);
}

reverseDiffpLR <- function(data, class, epsilon, missing, guess) {
	
	odata <- data;
	data <- as.matrix(data);
	dnames <- dimnames(data)[[2]];
	
	missmeans <- numeric(0);
	for(name in missing) {
		missmeans <- append(missmeans, mean(data[,name]));
	}
	
	norm <- normalize(data, class);
	data <- norm$data;
	
	trainx <- data[,!(dimnames(data)[[2]] %in% class)];
	trainx <- cbind(as.matrix(trainx), numeric(nrow(trainx))+1);
	
	trainy <- data[,class];
	
	d <- length(dimnames(data)[[2]]);
	
	r0 <- t(trainx) %*% trainx;
	r1 <- -2 * (t(trainx) %*% trainy);
	
	sensitivity <- 2*d*d + 4*d;
	
	noisematrix1 <- NA;
	noisematrix2 <- NA;
	if(epsilon <= 0) {
		noisematrix1 <- matrix(0, nrow = d, ncol = d);
		noisematrix2 <- numeric(d);
	} else {
		noisematrix1 <- matrix(laprnd(d*d, 0, sensitivity * 1/epsilon), nrow = d, ncol = d);
		noisematrix2 <- laprnd(d, 0, sensitivity * 1/epsilon);
	}
	
	coe2 <- r0 + noisematrix1;
	coe2 <- 0.5*(t(coe2) + coe2);
	if(epsilon > 0) {
		coe2 <- coe2 + 4 * sqrt(2) * sensitivity * (1/epsilon) * diag(d);
	} else {
		#coe2 <- coe2 + 10 * sqrt(2) * sensitivity * (1/20) * diag(d);
	}
	coe1 <- r1 + noisematrix2;
			
	eigs <- eigen(coe2);
	vec <- t(apply(eigs$vectors, 1, rev));
	val <- rev(eigs$values);	
		
	del <- val >= 1e-8;
	val <- diag(val);
	val <- val[del,];
	val <- val[,del];
		
	vec <- vec[,del];		
	
	coe2 <- val;
	coe1 <- t(vec) %*% coe1;
	
	noiseobj <- function(w) {
		return(t(w) %*% coe2 %*% w + t(coe1) %*% w);
	}
	
	g0 <- runif(d-sum(!del), 0, 1);
		
	optrsol <- optim(g0, noiseobj, method = "BFGS", control = list(maxit=100000));
	
	g <- optrsol$par;
	
	bestw <- vec %*% g;
	
	w <- bestw[1:length(bestw)-1];
	b <- bestw[length(bestw)];
	
	trueval <- norm$dtransform(numeric(length(data[1,]))+1)[1];
	
	nonclassnames <- dnames[!(dnames %in% class)];
	nonguessnames <- dnames[!(dnames %in% guess) & !(dnames %in% class)];
	knownnames <- nonclassnames[!(nonclassnames %in% missing) & !(nonclassnames %in% guess)];
	
	pdfs <- getDatasetPDFs(data[,missing], 2, length(missing));
	dataprobs <- data[,missing];
	rowprobs <- apply(as.array(1:nrow(data)), 1, function(x) getSampleProbability(data[x,missing], length(missing), pdfs));
	dataprobs <- cbind(dataprobs, rowprobs);
	colnames(dataprobs) <- append(missing, "class");
	pdfpredict <- rawLRNoNormalize(dataprobs, "class", 0);	
		
	unqdata <- lapply(missing, function(name) unique(data[,name]));
	names(unqdata) <- missing;
	missingcombs <- unique(fixupVectorAttributes(expand.grid(unqdata), getVectorAttributes(missing), trueval));
	row.names(missingcombs) <- 1:nrow(missingcombs);			
	
	fmls <- apply(as.array(guess), 1, function(name) sprintf("xm[,%d] <= 0", match(name, dnames)));
	fmla <- paste(fmls, collapse=" & ");
		
	resids <- data[,class] - (b + data[,nonclassnames] %*% w);
	errmean <- as.numeric(mean(resids));
	errvar <- as.numeric(var(resids));
	errsd <- as.numeric(sd(resids));
	errscale <- sqrt(errvar/2);
		
	gprobs <- vectorAttrProbs(odata, guess);	
		
	clfun <- cmpfun(function(x) {
								
		xorig <- x;
		x[guess] <- 0;				
				
		x <- norm$dtransform(as.numeric(x));
		xorig.transf <- norm$dtransform(as.numeric(xorig));
		if(length(x) == d) {		
			dose <- x[match(class, dnames)];
			#dose <- as.numeric(norm$dtransform(WarfDose(norm$runtransform(as.numeric(b + xorig.transf[match(nonclassnames, dnames)] %*% w)))));
			x <- x[match(nonclassnames, dnames)];			
		}
		#dose <- as.numeric(b + x %*% w);
		
		known <- x[match(knownnames, nonclassnames)];
		known <- matrix(data=known, nrow=nrow(missingcombs), ncol=length(known), byrow=TRUE);		
		colnames(known) <- knownnames;
		missingcombs <- cbind(missingcombs, known);
		missingcombs <- missingcombs[,nonclassnames[!(nonclassnames %in% guess)]];		
	
		xm <- matrix(data=x, nrow=length(guess)+1, ncol=length(x), byrow=TRUE);
		for(name in guess) xm[match(name, guess)+1, match(name, dnames)] <- trueval;	
				
		xmnonmiss <- NA;
		if(length(guess) <= 1) {
			xmnonmiss <- data.frame(xm[,match(guess, nonclassnames)]);
			names(xmnonmiss) <- guess;		
			xmnonmiss <- as.matrix(xmnonmiss);
		} else {
			xmnonmiss <- xm[,match(guess, nonclassnames)];
			colnames(xmnonmiss) <- guess;					
		}
		
		recon <- merge(missingcombs, xmnonmiss);
		xm <- as.matrix(recon[nonclassnames]);
		#print(xm);
		
		probs <- pdfpredict$b + xm[,missing] %*% pdfpredict$w; #apply(as.array(1:nrow(xm)), 1, function(j) getSampleProbability(xm[j,missing], length(missing), pdfs));
		probsneg <- which(probs < 0);
		probs[probsneg] <- 0;
		finalprobs <- exp(log(probs) + apply(as.array(1:nrow(xm)), 1, function(i) log(gprobs[getVectorAttrVal(xm[i,guess],guess)+1])) + dnorm(dose - (b + xm %*% w), 0, errsd, log=TRUE));
		
		whichwild <- which(eval(parse(text=fmla)));
		totmass <- sum(finalprobs[whichwild]);
		totmass <- append(totmass, apply(as.array(guess), 1,
			function(name) sum(finalprobs[which(xm[,match(name, dnames)] > 0)])
		));
		winner <- which.max(totmass)-1;
		return(winner);
		
		#rvect <- abs(dose - (b + xm %*% w)) * (1 - probs);
		
		#winner <- xm[which.min(rvect),];
		#names(winner) <- dnames[dnames != class];
		
		#return(getVectorAttrVal(winner, guess));						
	});
	
	return(clfun);
}

reverseDiffpLRNoMissing <- function(data, class, epsilon, guess) {
	
	odata <- data;
	data <- as.matrix(data);
	dnames <- dimnames(data)[[2]];
		
	norm <- normalize(data, class);
	data <- norm$data;
	
	trainx <- data[,!(dimnames(data)[[2]] %in% class)];
	trainx <- cbind(as.matrix(trainx), numeric(nrow(trainx))+1);
	
	trainy <- data[,class];
	
	d <- length(dimnames(data)[[2]]);
	
	r0 <- t(trainx) %*% trainx;
	r1 <- -2 * (t(trainx) %*% trainy);
	
	sensitivity <- 2*d*d + 4*d;
	
	noisematrix1 <- NA;
	noisematrix2 <- NA;
	if(epsilon <= 0) {
		noisematrix1 <- matrix(0, nrow = d, ncol = d);
		noisematrix2 <- numeric(d);
	} else {
		noisematrix1 <- matrix(laprnd(d*d, 0, sensitivity * 1/epsilon), nrow = d, ncol = d);
		noisematrix2 <- laprnd(d, 0, sensitivity * 1/epsilon);
	}
	
	coe2 <- r0 + noisematrix1;
	coe2 <- 0.5*(t(coe2) + coe2);

	coe1 <- r1 + noisematrix2;

	if(epsilon > 0) {
		coe2 <- coe2 + 4 * sqrt(2) * sensitivity * (1/epsilon) * diag(d);							
	} else {
		#coe2 <- coe2 + 10 * sqrt(2) * sensitivity * (1/20) * diag(d);
	}
	
	eigs <- eigen(coe2);
	vec <- t(apply(eigs$vectors, 1, rev));
	val <- rev(eigs$values);	
		
	del <- val >= 1e-8;
	val <- diag(val);
	val <- val[del,];
	val <- val[,del];
		
	vec <- vec[,del];		
	
	coe2 <- val;
	coe1 <- t(vec) %*% coe1;
	
	noiseobj <- function(w) {
		return(t(w) %*% coe2 %*% w + t(coe1) %*% w);
	}
	
	g0 <- runif(d-sum(!del), 0, 1);
		
	optrsol <- optim(g0, noiseobj, method = "BFGS", control = list(maxit=100000));
	
	g <- optrsol$par;
	
	bestw <- vec %*% g;
	
	w <- bestw[1:length(bestw)-1];
	b <- bestw[length(bestw)];
	
	trueval <- norm$dtransform(numeric(length(data[1,]))+1)[1];
	
	nonclassnames <- dnames[!(dnames %in% class)];
	nonguessnames <- dnames[!(dnames %in% guess) & !(dnames %in% class)];
	knownnames <- nonclassnames[!(nonclassnames %in% guess)];
			
	fmls <- apply(as.array(guess), 1, function(name) sprintf("xm[,%d] <= 0", match(name,nonclassnames)));
	fmla <- paste(fmls, collapse=" & ");
		
	resids <- data[,class] - (b + data[,nonclassnames] %*% w);
	errmean <- as.numeric(mean(resids));
	errvar <- as.numeric(var(resids));
	errsd <- as.numeric(sd(resids));
	errscale <- sqrt(errvar/2);
	
	gprobs <- vectorAttrProbs(odata, guess);	
	
	dosemean <- as.numeric(mean(odata[,"dose"]));
		
	clfun <- cmpfun(function(x) {
				
		xorig <- x;				
		x[guess] <- 0;		
								
		xorig.transf <- norm$dtransform(as.numeric(xorig));
		#x[match(class, dnames)] <- WarfDose(norm$runtransform(as.numeric(b + xorig.transf[match(nonclassnames, dnames)] %*% w)));
		x <- norm$dtransform(as.numeric(x));
		if(length(x) == d) {
			dose <- x[match(class, dnames)];	
			#dose <- as.numeric(b + xorig.transf[match(nonclassnames, dnames)] %*% w);
			x <- x[match(nonclassnames, dnames)];
		}
		
		known <- x[match(knownnames, nonclassnames)];
		names(known) <- knownnames;
	
		xm <- matrix(data=x, nrow=length(guess)+1, ncol=length(x), byrow=TRUE);
		for(name in guess) xm[match(name, guess)+1, match(name, dnames)] <- trueval;	
				
		xmnonmiss <- NA;
		if(length(guess) <= 1) {
			xmnonmiss <- data.frame(xm[,match(guess, nonclassnames)]);
			names(xmnonmiss) <- guess;		
			xmnonmiss <- as.matrix(xmnonmiss);
		} else {
			xmnonmiss <- xm[,match(guess, nonclassnames)];
			colnames(xmnonmiss) <- guess;					
		}
		
		recon <- merge(t(as.matrix(known)), xmnonmiss);
		xm <- as.matrix(recon[nonclassnames]);
				
		#finalprobs <- exp(apply(as.array(1:nrow(xm)), 1, function(i) log(gprobs[getVectorAttrVal(xm[i,guess],guess)+1])) + dnorm(dose - (b + xm %*% w), 0, errsd, log=TRUE));
		finalprobs <- dnorm(dose - (b + xm %*% w), 0, errsd, log=FALSE);
		
		whichwild <- which(eval(parse(text=fmla)));
		totmass <- sum(finalprobs[whichwild]);
		totmass <- append(totmass, apply(as.array(guess), 1,
			function(name) sum(finalprobs[which(xm[,match(name, dnames)] > 0)])
		));
		winner <- which.max(totmass)-1;
		return(winner);
		
		#rvect <- abs(dose - (b + xm %*% w)) * (1 - probs);
		
		#winner <- xm[which.min(rvect),];
		#names(winner) <- dnames[dnames != class];
		
		#return(getVectorAttrVal(winner, guess));						
	});
	
	return(clfun);
}

getAttributeEntropy <- function(data, validation, epsilon, class, missing, guess) {
	
	data <- as.matrix(data);
	validation <- as.matrix(validation);
	
	rawmodel <- rawLR(data, class, epsilon);
	w <- rawmodel$w;
	b <- rawmodel$b;
	data <- rawmodel$data;
	transform <- rawmodel$dtransform;
	
	trueval <- max(data[data[,guess[1]] > 0,guess[1]]);
	
	dnames <- dimnames(data)[[2]];
	
	nonclassnames <- dnames[!(dnames %in% class)];
	knownnames <- nonclassnames[!(nonclassnames %in% missing) & !(nonclassnames %in% guess)];
	unknownnames <- append(nonclassnames[!(nonclassnames %in% knownnames) & !(nonclassnames %in% guess)], guess);	
	
	resids <- data[,class] - (b + data[,nonclassnames] %*% w);
	errmean <- as.numeric(mean(resids));
	errvar <- as.numeric(var(resids));	
	errscale <- as.numeric(sqrt(errvar/2));
	
	pdfs <- getDatasetPDFs(data[,unknownnames], 5, length(unknownnames));
	dataprobs <- data[,unknownnames];
	dataprobs <- cbind(dataprobs, apply(as.array(1:nrow(data)), 1, function(x) getSampleProbability(data[x,unknownnames], length(unknownnames), pdfs)));
	colnames(dataprobs) <- append(unknownnames, "class");
	pdfpredict <- rawLR(dataprobs, "class", 0);	
	
	unqdata <- lapply(missing, function(name) unique(data[,name]));
	names(unqdata) <- missing;
	missingcombs <- unique(fixupVectorAttributes(expand.grid(unqdata), getVectorAttributes(missing), trueval));
	row.names(missingcombs) <- 1:nrow(missingcombs);
	
	fmls <- apply(as.array(guess), 1, function(name) sprintf("xm[,%d] == 0", match(name, dnames)));
	fmla <- paste(fmls, collapse=" & ");
	
	entropies <- numeric(0);
	
	baseentropy <- entropy(numeric(length(guess)+1)+1/(length(guess)+1));
	
	for(i in 1:nrow(validation)) {
		
		x <- transform(validation[i,]);
		x[guess] <- 0.0;
				
		xm <- matrix(data=x, nrow=length(guess)+1, ncol=length(x), byrow=TRUE);
		for(name in guess) xm[match(name, guess)+1, match(name, dnames)] <- trueval;		
		
		xmnonmiss <- xm[,match(guess, nonclassnames)];
		colnames(xmnonmiss) <- guess;
		
		xm <- as.matrix(merge(missingcombs, xmnonmiss));
		#print(xm);
		
		probs <- apply(as.array(1:nrow(xm)), 1, function(j) getSampleProbability(xm[j,], length(unknownnames), pdfs));
		#probs <- pdfpredict$b + xm %*% pdfpredict$w;		
		
		known <- x[match(knownnames, nonclassnames)];
		known <- matrix(data=known, nrow=nrow(missingcombs), ncol=length(known), byrow=TRUE);
		colnames(known) <- knownnames;
		umissingcombs <- cbind(missingcombs, known);
		umissingcombs <- umissingcombs[,nonclassnames[!(nonclassnames %in% guess)]];

		recon <- merge(umissingcombs, xmnonmiss);
		xm <- as.matrix(recon[nonclassnames]);
		
		#print(x[class] - (b + xm %*% w));
		
		probs <- exp(probs*dlaplace(x[class] - (b + xm %*% w), location=errmean, scale=errscale, log=FALSE));
		probs <- probs / sum(probs);
		
		totmass <- sum(probs[which(eval(parse(text=fmla)))]);
		totmass <- append(totmass, apply(as.array(guess), 1,
			function(name) sum(probs[which(xm[,match(name, dnames)] > 0)])));
				
		entropies <- append(entropies, entropy(totmass));
				
		#if(i %% 100 == 0) cat(mean(entropies), " ");
	}
	
	return(mean(entropies));
}

entropy <- function(p) {
	
	return(-1*sum(apply(as.array(p), 1, function(x) ifelse(x == 0, 0, x*log(x)))));
}

getMostLikelyRows <- function(training, class, n) {
		
	dnames <- dimnames(training)[[2]];
	clparams <- rawLR(training, class, 0);
	data <- numericFromNominal(discretizeDataset(clparams$data, 2));
	# assume the first column is binary-valued
	trueval <- as.numeric(data[which(data[,1] > 0)[1],1][1]);
	
	pdfs <- getDatasetPDFs(data, 2, ncol(data));
	dataprobs <- apply(as.array(1:nrow(data)), 1, function(x) getSampleProbability(data[x,], ncol(data), pdfs));
	
	unqdata <- lapply(dnames, function(name) unique(data[,name]));
	names(unqdata) <- dnames;
	allcombs <- unique(fixupVectorAttributes(expand.grid(unqdata), getVectorAttributes(dnames), trueval));
	#row.names(allcombs) <- 1:nrow(allcombs);
	
	return(allcombs[order(dataprobs, decreasing=TRUE)[1:n],]);	
}

getDatasetFromLRClassifier <- function(training, class, epsilon, allcombs, initpoint) {
	
	clparams <- rawLR(training, class, epsilon);
	training <- clparams$data;
	
	dnames <- dimnames(training)[[2]];
	
	allcombs <- as.matrix(allcombs);
	
	#pdfs <- getDatasetPDFs(allcombs, 2, ncol(allcombs));
	#params0 <- apply(as.array(1:nrow(allcombs)), 1, function(x) getSampleProbability(allcombs[x,], ncol(allcombs), pdfs));		
	#params0 <- numeric(nrow(allcombs))+0.5; #runif(nrow(allcombs),0,1);	
	params0 <- initpoint;
	realval <- append(clparams$w, clparams$b);
	
	obj <- function(w) {
		sdat <- allcombs[sample.int(n=nrow(allcombs), size=1000, replace=TRUE, prob=abs(w)),];
		np <- (apply(as.array(1:1), 1, function(i) { lrr <- rawLRNoNormalize(sdat, class, epsilon); append(lrr$w, lrr$b); }));
		#print(np);
		
		rv <- sqrt(sum((np-realval)^2));
		cat(rv, " ");
		return(rv);
		
		#print(sqrt(sum(sweep(np, 2, realval)^2))/1);
		#return(sqrt(sum(sweep(np, 2, realval)^2))/1);
	}
		
	#optrsol <- optim(params0, obj, method="L-BFGS-B", lower=numeric(nrow(allcombs)), upper=numeric(nrow(allcombs))+1, control = list(maxit=10000));
	optrsol <- psoptim(params0, obj, control = list(maxit=10000));
	
	probs <- optrsol$par;
	guessdata <- allcombs[sample.int(n=nrow(allcombs), size=1000, replace=TRUE, prob=abs(probs)),];
	redata <- t(apply(as.array(1:nrow(guessdata)), 1, function(x) rawclfun$untransform(as.numeric(guessdata[x,]))));
	colnames(redata) <- dnames;
	
	return(redata);
}

getMultipleClassifierDatasets <- function(training, class, allcombs, params, n) {
	
	for(eps in params) {
		for(i in 1:n) {
			print(sprintf("cldata%.2f-%d.arff", eps, i));
			cdata <- getDatasetFromLRClassifier(training, class, eps, allcombs);
			write.arff(cdata, sprintf("cldata%.2f-%d.arff", eps, i));
		}
	}
}

getPDF <- function(data, ncut) {
	
	data <- data.frame(data);
	data <- discretizeDataset(data, ncut);
	
	ptable <- prop.table(table(data));
	
	xs <- as.numeric(names(ptable));
	ys <- apply(as.array(names(ptable)), 1, function(name) ptable[name]);
	ys <- append(ys, ys[length(ys)]);
		
	return(stepfun(xs, ys, right=TRUE));
}

getDatasetPDFs <- function(data, ncut, ncols) {
	
	data <- as.matrix(data);
	dnames <- dimnames(data)[[2]];
	
	ret <- list();
	
	for(name in dnames) {
		
		col <- data[,name];
		efncut <- ncut;
		if(length(unique(col)) < ncut) efncut <- length(unique(col));
		
		ret[[match(name, dnames)]] <- getPDF(col, efncut);		
	}
	
	probs <- apply(as.array(1:nrow(data)), 1, function(i) exp(sum(apply(as.array(1:ncols), 1, function(x) log(ret[[x]](data[i,x]))))));
	maxprob <- max(probs);
	
	return(list(pdfs=ret, max=maxprob));
}

getSampleProbability <- function(sample, ncols, pdfs) {
	
	return(exp(sum(apply(as.array(1:ncols), 1, function(x) log(pdfs$pdfs[[x]](sample[x])))))/pdfs$max);
	#return(prod(apply(as.array(1:ncols), 1, function(x) pdfs$pdfs[[x]](sample[x]))))/pdfs$max;
}

vectorAttr <- function(attr) {
	if(attr == "vkorc1") {
		return(c("vkorc1=CT", "vkorc1=TT"));
	} else {
		return(c("cyp2c9=13", "cyp2c9=12", "cyp2c9=23", "cyp2c9=33", "cyp2c9=22"));
	}
}

t.test.silent <- function(...) { 
	obj<-try(t.test(...), silent=TRUE) 
	if (is(obj, "try-error")) return(NA) else return(obj$p.value); 
}

getTestStatistics <- function(trainstats, validstats, baseline, baselinediff) {
		
	gt.95 <- t.test.silent(x=trainstats, y=validstats, alternative="g", conf.level=0.95);
	tt.95 <- t.test.silent(x=trainstats, y=validstats, alternative="t", conf.level=0.95);
	baseline.95 <- t.test.silent(x=trainstats-baseline, alternative="g", conf.level=0.95);
	baselinediff.95 <- t.test.silent(x=baselinediff-(trainstats-validstats), alternative="g", conf.level=0.95);
	
	summ <- sprintf("gt.95=%f tt.95=%f base.95=%f diff.95=%f", gt.95, tt.95, baseline.95, baselinediff.95);
	
	return(list(gt.95=gt.95, tt.95=tt.95, baseline.95=baseline.95, baselinediff.95=baselinediff.95, summ=summ));
}

baseline.acc <- function(data, guess) {
	perfs <- apply(as.array(guess), 1, function(name) mean(data[,name]));
	if(sum(perfs) < 0.5) return(max(perfs));
	
	return(1-sum(perfs));
}

baseline.auc <- function(data, guess) {
	return(0.5);
}

runMultipleReverseClassifiers <- function(training, validation, class, eps, missing, guess, nruns, statsfun) {
	
	accs <- numeric(0);
	auc <- numeric(0);
	vaccs <- numeric(0);
	vauc <- numeric(0);			
		
	set.seed(1);
	
	if(eps==0) nruns <- 1;
	
	rclfun <- NULL;
	if(length(missing)==0) {
		rclfun <- reverseDiffpLRNoMissing(training, class, 0, guess);
	} else {
		rclfun <- reverseDiffpLR(training, class, 0, missing, guess);
	}
	results <- statsfun(rclfun, training, guess);
	vresults <- statsfun(rclfun, validation, guess);
	baselineacc <- baseline.acc(training, guess);
	baselineauc <- baseline.auc(training, guess);
	bdiffacc <- mean(results$acc)-mean(vresults$acc);
	bdiffauc <- results$auc-vresults$auc;
	
	print(sprintf("blineacc: %f, blineauc: %f, bdiffacc: %f (%f - %f), bdiffauc: %f (%f - %f)", baselineacc, baselineauc, bdiffacc, mean(results$acc), mean(vresults$acc), bdiffauc, mean(results$auc), mean(vresults$auc)));
	
	for(i in 1:nruns) {
		
		rclfun <- NULL;
		if(length(missing)==0) {
			rclfun <- try(reverseDiffpLRNoMissing(training, class, eps, guess), silent=TRUE);
		} else {
			rclfun <- try(reverseDiffpLR(training, class, eps, missing, guess), silent=TRUE);
		}
		if (is(rclfun, "try-error")) next;
		nresults <- statsfun(rclfun, training, guess);
		nvresults <- statsfun(rclfun, validation, guess);

		accs <- append(accs, mean(nresults$acc));
		auc <- append(auc, nresults$auc);
		vaccs <- append(vaccs, mean(nvresults$acc));
		vauc <- append(vauc, nvresults$auc);
		
		if(i > 1) {

			accstats <- getTestStatistics(accs, vaccs, baselineacc, bdiffacc);
			aucstats <- getTestStatistics(auc, vauc, baselineauc, bdiffauc);

			print(sprintf("eps = %f, i = %d, train: (acc: %f, auc: %f) valid: (acc: %f, auc: %f) diff: (acc: %f, auc: %f)", eps, i, mean(accs), mean(auc), mean(vaccs), mean(vauc), mean(accs)-mean(vaccs), mean(auc)-mean(vauc)));
			print(sprintf("  acc p: %s", accstats$summ));
			print(sprintf("  auc p: %s", aucstats$summ));
		} else {
			print(sprintf("train: (acc: %f p = %f, auc: %f p = %f) valid: (acc: %f, auc: %f)", mean(accs), Inf, mean(auc), Inf, mean(vaccs), mean(vauc)));
		}
	}	
	
	summ <- list();
	if(nruns > 1) {
		
		accstats <- getTestStatistics(accs, vaccs, baselineacc, bdiffacc);
		aucstats <- getTestStatistics(auc, vauc, baselineauc, bdiffauc);
		
		summ$base <- sprintf("train: (acc: %f, auc: %f) valid: (acc: %f, auc: %f) diff: (acc: %f, auc: %f)", mean(accs), mean(auc), mean(vaccs), mean(vauc), mean(accs)-mean(vaccs), mean(auc)-mean(vauc));
		summ$accp <- sprintf("  acc p: %s", accstats$summ);
		summ$aucp <- sprintf("  auc p: %s", aucstats$summ);
	} else {
		summ <- sprintf("train: (acc: %f p = %f, auc: %f p = %f) valid: (acc: %f, auc: %f)", mean(accs), Inf, mean(auc), Inf, mean(vaccs), mean(vauc));
	}
	
	curdir <- getwd();
	dir.create(file.path(curdir, "privacy-results"), showWarnings = FALSE);
	setwd(file.path(curdir, "privacy-results"));
	filename <- sprintf("runMultipleReverseClassifiers-eps.%.2f-missing.%d-guess.%s-nruns.%d-statsfun.%s-%s.Robject", eps, length(missing), strsplit(guess[1],"=")[[1]][1], nruns, deparse(substitute(statsfun)), format(Sys.time(), "%b%d.%H.%M.%S"));
	dput(list(summary=summ, baseline = sprintf("blineacc: %f, blineauc: %f, bdiffacc: %f (%f - %f), bdiffauc: %f (%f - %f)", baselineacc, baselineauc, bdiffacc, mean(results$acc), mean(vresults$acc), bdiffauc, results$auc, vresults$auc), accs=accs, auc=auc, vaccs=vaccs, vauc=vauc), file=filename);
	setwd(curdir);
	
	return(list(summary=summ, baseline = sprintf("blineacc: %f, blineauc: %f, bdiffacc: %f (%f - %f), bdiffauc: %f (%f - %f)", baselineacc, baselineauc, bdiffacc, mean(results$acc), mean(vresults$acc), bdiffauc, results$auc, vresults$auc), accs=accs, auc=auc, vaccs=vaccs, vauc=vauc));		
}

runMultipleReverseClassifiersDifferentSizes <- function(training, validation, class, eps, missing, guess, nruns, statsfun) {
	
	endresult <- list();
	
	accs <- numeric(0);
	fmeas <- numeric(0);
	auc <- numeric(0);
	vaccs <- numeric(0);
	vfmeas <- numeric(0);
	vauc <- numeric(0);		

	rclfun <- NULL;
	if(length(missing)==0) {
		rclfun <- reverseDiffpLRNoMissing(training, class, 0, guess);
	} else {
		rclfun <- reverseDiffpLR(training, class, 0, missing, guess);
	}
	results <- statsfun(rclfun, training, guess);
	vresults <- statsfun(rclfun, validation, guess);
	baselineacc <- baseline.acc(training, guess);
	baselineauc <- baseline.auc(training, guess);
	bdiffacc <- mean(results$acc)-mean(vresults$acc);
	bdiffauc <- results$auc-vresults$auc;
	
	print(sprintf("blineacc: %f, blineauc: %f, bdiffacc: %f, bdiffauc: %f", baselineacc, baselineauc, bdiffacc, bdiffauc));
		
	set.seed(1);
	
	for(div in seq(0.01,1,by=0.2)) {
		
		accs <- numeric(0);
		fmeas <- numeric(0);
		auc <- numeric(0);
		vaccs <- numeric(0);
		vfmeas <- numeric(0);
		vauc <- numeric(0);				
	
		for(i in 1:nruns) {						
			
			tfrac <- training[sample.int(nrow(training), size=round(nrow(training)*div)),];			
			
			rclfun <- NULL;
			if(length(missing) == 0) {
				rclfun <- try(reverseDiffpLRNoMissing(tfrac, class, eps, guess), silent=TRUE);
			} else {
				rclfun <- try(reverseDiffpLR(tfrac, class, eps, missing, guess), silent=TRUE);
			}
			if (is(rclfun, "try-error")) next;
			nresults <- statsfun(rclfun, tfrac, guess);
			nvresults <- statsfun(rclfun, validation, guess);
			
			accs <- append(accs, mean(nresults$acc));
			fmeas <- append(fmeas, nresults$fmeas);
			auc <- append(auc, nresults$auc);
			vaccs <- append(vaccs, mean(nvresults$acc));
			vfmeas <- append(vfmeas, nvresults$fmeas);
			vauc <- append(vauc, nvresults$auc);
			
			if(i > 1) {
	
				accstats <- getTestStatistics(accs, vaccs, baselineacc, bdiffacc);
				aucstats <- getTestStatistics(auc, vauc, baselineauc, bdiffauc);
	
				print(sprintf("i = %d, train: (acc: %f, auc: %f) valid: (acc: %f, auc: %f) diff: (acc: %f, auc: %f)", i, mean(accs), mean(auc), mean(vaccs), mean(vauc), mean(accs)-mean(vaccs), mean(auc)-mean(vauc)));
				print(sprintf("  acc p: %s", accstats$summ));
				print(sprintf("  auc p: %s", aucstats$summ));
			} else {
				print(sprintf("train: (acc: %f p = %f, auc: %f p = %f) valid: (acc: %f, auc: %f)", mean(accs), Inf, mean(auc), Inf, mean(vaccs), mean(vauc)));
			}
		}
		
		print("-----------------------");
		
		endresult[[as.character(div)]] <- list();
		if(nruns > 1) {
			
			accstats <- getTestStatistics(accs, vaccs, baselineacc, bdiffacc);
			aucstats <- getTestStatistics(auc, vauc, baselineauc, bdiffauc);
			
			endresult[[as.character(div)]]$base <- sprintf("train: (acc: %f, auc: %f) valid: (acc: %f, auc: %f) diff: (acc: %f, auc: %f)", mean(accs), mean(auc), mean(vaccs), mean(vauc), mean(accs)-mean(vaccs), mean(auc)-mean(vauc));
			endresult[[as.character(div)]]$accs <- accs;
			endresult[[as.character(div)]]$auc <- auc;
			endresult[[as.character(div)]]$vaccs <- vaccs;
			endresult[[as.character(div)]]$vauc <- vauc;
			endresult[[as.character(div)]]$accp <- sprintf("  acc p: %s", accstats$summ);
			endresult[[as.character(div)]]$aucp <- sprintf("  auc p: %s", aucstats$summ);
		} else {
			endresult[[as.character(div)]] <- sprintf("train: (acc: %f p = %f, auc: %f p = %f) valid: (acc: %f, auc: %f)", mean(accs), Inf, mean(auc), Inf, mean(vaccs), mean(vauc));
		}
		
	}		
	
	curdir <- getwd();
	dir.create(file.path(curdir, "privacy-results"), showWarnings = FALSE);
	setwd(file.path(curdir, "privacy-results"));
	filename <- sprintf("runMultipleReverseClassifiersWithDifferentSizes-eps.%.2f-missing.%d-guess.%s-nruns.%d-statsfun.%s-%s.Robject", eps, length(missing), strsplit(guess[1],"=")[[1]][1], nruns, deparse(substitute(statsfun)), format(Sys.time(), "%b%d.%H.%M.%S"));
	dput(list(endresult=endresult, baseline = sprintf("blineacc: %f, blineauc: %f, bdiffacc: %f (%f - %f), bdiffauc: %f (%f - %f)", baselineacc, baselineauc, bdiffacc, mean(results$acc), mean(vresults$acc), bdiffauc, results$auc, vresults$auc)), file=filename);
	setwd(curdir);
	
	return(list(endresult=endresult, baseline = sprintf("blineacc: %f, blineauc: %f, bdiffacc: %f (%f - %f), bdiffauc: %f (%f - %f)", baselineacc, baselineauc, bdiffacc, mean(results$acc), mean(vresults$acc), bdiffauc, results$auc, vresults$auc)));	
	
	return(endresult);
}


runMultipleReverseClassifiersKFolds <- function(training, class, eps, missing, guess, nruns, K, statsfun) {
	
	accs <- numeric(0);
	auc <- numeric(0);
	vaccs <- numeric(0);
	vauc <- numeric(0);		

	rclfun <- NULL;
	if(length(missing)==0) {
		rclfun <- reverseDiffpLRNoMissing(training, class, 0, guess);
	} else {
		rclfun <- reverseDiffpLR(training, class, 0, missing, guess);
	}
	results <- statsfun(rclfun, training, guess);
	vresults <- statsfun(rclfun, validation, guess);
	baselineacc <- baseline.acc(training, guess);
	baselineauc <- baseline.auc(training, guess);
	bdiffacc <- mean(results$acc)-mean(vresults$acc);
	bdiffauc <- results$auc-vresults$auc;
	
	print(sprintf("blineacc: %f, blineauc: %f, bdiffacc: %f, bdiffauc: %f", baselineacc, baselineauc, bdiffacc, bdiffauc));		
		
	set.seed(1);
	
	foldresults <- list();	
	foldstore <- list();
	for(k in 1:K) {
		foldstore[[k]] <- list(accs=accs, auc=auc, vaccs=vaccs, vauc=vauc);
	}

	for(i in 1:nruns) {	
		
		folds <- genfolds(training, K);
		
		for(k in 1:K) {
			
			#accs <- foldstore[[k]]$accs;
			#vaccs <- foldstore[[k]]$vaccs;
			#auc <- foldstore[[k]]$auc;
			#vauc <- foldstore[[k]]$vauc;		

			train <- na.omit(folds[[k]]$train);
			valid <- na.omit(folds[[k]]$test);
			
			rclfun <- NULL;
			if(length(missing) == 0) {
				rclfun <- try(reverseDiffpLRNoMissing(train, class, eps, guess), silent=TRUE);
			} else {
				rclfun <- try(reverseDiffpLR(train, class, eps, missing, guess), silent=TRUE);
			}
			if (is(rclfun, "try-error")) next;
			nresults <- statsfun(rclfun, train, guess);
			nvresults <- statsfun(rclfun, valid, guess);
			
			#cat(paste("( acc=", signif(results$acc, 3), "fmeas=", signif(results$fmeas, 3), "auc=", signif(results$auc, 3), ")"));		
			
			accs <- append(accs, mean(nresults$acc));
			auc <- append(auc, nresults$auc);
			vaccs <- append(vaccs, mean(nvresults$acc));
			vauc <- append(vauc, nvresults$auc);
			
			if(i > 1 || k > 1) {
	
				accstats <- getTestStatistics(accs, vaccs, baselineacc, bdiffacc);
				aucstats <- getTestStatistics(auc, vauc, baselineauc, bdiffauc);
	
				print(sprintf("runMultipleReverseClassifiersKFolds-eps.%.2f-missing.%d-guess.%s-nruns.%d-statsfun.%s-%s.Robject", eps, length(missing), strsplit(guess[1],"=")[[1]][1], nruns, deparse(substitute(statsfun)), format(Sys.time(), "%b%d.%H.%M.%S")));
				print(sprintf("i = %d, k=%d, train: (acc: %f, auc: %f) valid: (acc: %f, auc: %f) diff: (acc: %f, auc: %f)", i, k, mean(accs), mean(auc), mean(vaccs), mean(vauc), mean(accs)-mean(vaccs), mean(auc)-mean(vauc)));
				print(sprintf("  acc p: %s", accstats$summ));
				print(sprintf("  auc p: %s", aucstats$summ));
			} else {
				print(sprintf("k = %d, train: (acc: %f p = %f, auc: %f p = %f) valid: (acc: %f, auc: %f)", k, mean(accs), Inf, mean(auc), Inf, mean(vaccs), mean(vauc)));
			}			
			#cat(paste(signif(mean(accs), 3), " "))
			
			summ <- list();
			if(i > 1 || k > 1) {
				
				accstats <- getTestStatistics(accs, vaccs, baselineacc, bdiffacc);
				aucstats <- getTestStatistics(auc, vauc, baselineauc, bdiffauc);
				
				summ$base <- sprintf("k = %d, train: (acc: %f, auc: %f) valid: (acc: %f, auc: %f) diff: (acc: %f, auc: %f)", k, mean(accs), mean(auc), mean(vaccs), mean(vauc), mean(accs)-mean(vaccs), mean(auc)-mean(vauc));
				summ$accp <- sprintf("  acc p: %s", accstats$summ);
				summ$aucp <- sprintf("  auc p: %s", aucstats$summ);
			} else {
				summ <- sprintf("k = %d, train: (acc: %f p = %f, auc: %f p = %f) valid: (acc: %f, auc: %f)", k, mean(accs), Inf, mean(auc), Inf, mean(vaccs), mean(vauc));
			}		
			
			foldresults <- list(summary=summ, baseline = sprintf("blineacc: %f, blineauc: %f, bdiffacc: %f (%f - %f), bdiffauc: %f (%f - %f)", baselineacc, baselineauc, bdiffacc, mean(results$acc), mean(vresults$acc), bdiffauc, results$auc, vresults$auc), accs=accs, auc=auc, vaccs=vaccs, vauc=vauc);
			
			#foldstore[[k]] <- list(accs=accs, auc=auc, vaccs=vaccs, vauc=vauc);
			
		}
		
	}
		
	curdir <- getwd();
	dir.create(file.path(curdir, "privacy-results"), showWarnings = FALSE);
	setwd(file.path(curdir, "privacy-results"));
	filename <- sprintf("runMultipleReverseClassifiersKFolds-eps.%.2f-missing.%d-guess.%s-nruns.%d-statsfun.%s-%s.Robject", eps, length(missing), strsplit(guess[1],"=")[[1]][1], nruns, deparse(substitute(statsfun)), format(Sys.time(), "%b%d.%H.%M.%S"));
	dput(foldresults, file=filename);
	setwd(curdir);
	
	return(foldresults);	
	
	return(summ);
}


runVkoReverseClassifiersWithDemo <- function(nruns) {
	vkovalid025 <- runMultipleReverseClassifiers(training, validation, "dose", 0.25, c("amiodarone", "decr", "cyp2c9=13", "cyp2c9=12", "cyp2c9=23", "cyp2c9=33", "cyp2c9=22"), c("vkorc1=CT", "vkorc1=TT"), nruns, rstatsrev);
	vkovalid1 <- runMultipleReverseClassifiers(training, validation, "dose", 1, c("amiodarone", "decr", "cyp2c9=13", "cyp2c9=12", "cyp2c9=23", "cyp2c9=33", "cyp2c9=22"), c("vkorc1=CT", "vkorc1=TT"), nruns, rstatsrev);
	vkovalid5 <- runMultipleReverseClassifiers(training, validation, "dose", 5, c("amiodarone", "decr", "cyp2c9=13", "cyp2c9=12", "cyp2c9=23", "cyp2c9=33", "cyp2c9=22"), c("vkorc1=CT", "vkorc1=TT"), nruns, rstatsrev);
	vkovalid20 <- runMultipleReverseClassifiers(training, validation, "dose", 20, c("amiodarone", "decr", "cyp2c9=13", "cyp2c9=12", "cyp2c9=23", "cyp2c9=33", "cyp2c9=22"), c("vkorc1=CT", "vkorc1=TT"), nruns, rstatsrev);
	vkovalid100 <- runMultipleReverseClassifiers(training, validation, "dose", 100, c("amiodarone", "decr", "cyp2c9=13", "cyp2c9=12", "cyp2c9=23", "cyp2c9=33", "cyp2c9=22"), c("vkorc1=CT", "vkorc1=TT"), nruns, rstatsrev);
}

runVkoReverseClassifiersWithAll <- function(nruns) {
	vkovalid025 <- runMultipleReverseClassifiers(training, validation, "dose", 0.25, c(), c("vkorc1=CT", "vkorc1=TT"), nruns, rstatsrev);
	vkovalid1 <- runMultipleReverseClassifiers(training, validation, "dose", 1, c(), c("vkorc1=CT", "vkorc1=TT"), nruns, rstatsrev);
	vkovalid5 <- runMultipleReverseClassifiers(training, validation, "dose", 5, c(), c("vkorc1=CT", "vkorc1=TT"), nruns, rstatsrev);
	vkovalid20 <- runMultipleReverseClassifiers(training, validation, "dose", 20, c(), c("vkorc1=CT", "vkorc1=TT"), nruns, rstatsrev);
	vkovalid100 <- runMultipleReverseClassifiers(training, validation, "dose", 100, c(), c("vkorc1=CT", "vkorc1=TT"), nruns, rstatsrev);
}

runCypReverseClassifiersWithDemo <- function(nruns) {	
	cypvalid025 <- runMultipleReverseClassifiers(training, validation, "dose", 0.25, c("amiodarone", "decr", "vkorc1=CT", "vkorc1=TT"), c("cyp2c9=13", "cyp2c9=12", "cyp2c9=23", "cyp2c9=33", "cyp2c9=22"), nruns, rstatsrev);
	cypvalid1 <- runMultipleReverseClassifiers(training, validation, "dose", 1, c("amiodarone", "decr", "vkorc1=CT", "vkorc1=TT"), c("cyp2c9=13", "cyp2c9=12", "cyp2c9=23", "cyp2c9=33", "cyp2c9=22"), nruns, rstatsrev);
	cypvalid5 <- runMultipleReverseClassifiers(training, validation, "dose", 5, c("amiodarone", "decr", "vkorc1=CT", "vkorc1=TT"), c("cyp2c9=13", "cyp2c9=12", "cyp2c9=23", "cyp2c9=33", "cyp2c9=22"), nruns, rstatsrev);
	cypvalid20 <- runMultipleReverseClassifiers(training, validation, "dose", 20, c("amiodarone", "decr", "vkorc1=CT", "vkorc1=TT"), c("cyp2c9=13", "cyp2c9=12", "cyp2c9=23", "cyp2c9=33", "cyp2c9=22"), nruns, rstatsrev);
	cypvalid100 <- runMultipleReverseClassifiers(training, validation, "dose", 100, c("amiodarone", "decr", "vkorc1=CT", "vkorc1=TT"), c("cyp2c9=13", "cyp2c9=12", "cyp2c9=23", "cyp2c9=33", "cyp2c9=22"), nruns, rstatsrev);
}

runCypReverseClassifiersWithAll <- function(nruns) {	
	cypvalid025 <- runMultipleReverseClassifiers(training, validation, "dose", 0.25, c(), c("cyp2c9=13", "cyp2c9=12", "cyp2c9=23", "cyp2c9=33", "cyp2c9=22"), nruns, rstatsrev);
	cypvalid1 <- runMultipleReverseClassifiers(training, validation, "dose", 1, c(), c("cyp2c9=13", "cyp2c9=12", "cyp2c9=23", "cyp2c9=33", "cyp2c9=22"), nruns, rstatsrev);
	cypvalid5 <- runMultipleReverseClassifiers(training, validation, "dose", 5, c(), c("cyp2c9=13", "cyp2c9=12", "cyp2c9=23", "cyp2c9=33", "cyp2c9=22"), nruns, rstatsrev);
	cypvalid20 <- runMultipleReverseClassifiers(training, validation, "dose", 20, c(), c("cyp2c9=13", "cyp2c9=12", "cyp2c9=23", "cyp2c9=33", "cyp2c9=22"), nruns, rstatsrev);
	cypvalid100 <- runMultipleReverseClassifiers(training, validation, "dose", 100, c(), c("cyp2c9=13", "cyp2c9=12", "cyp2c9=23", "cyp2c9=33", "cyp2c9=22"), nruns, rstatsrev);
}

getVectorAttributes <- function(x) {
	
	attrs <- list();
	names <- list();
	
	for(el in x) {
		splt <- strsplit(el, "=");
		if(length(splt[[1]]) == 1) next;
		
		name <- splt[[1]][1];
		if(!(name %in% names)) {
			names <- append(names, name);
			attrs[[match(name, names)]] <- character(0);
		}
		
		attrs[[match(name, names)]] <- append(attrs[[match(name, names)]], el);
	}
	
	return(attrs);
}

fixupVectorAttributes <- function(data, attrs, trueval) {
	
	if(length(attrs) <= 0) return(data);
	
	n <- nrow(data);
	
	for(i in 1:length(attrs)) {		
		
		cattr <- attrs[[i]];
		probs <- apply(as.array(cattr), 1, function(name) sum(data[,name])/n);
		if(sum(probs) >= 1) {
			probs <- numeric(length(cattr)+1) + 1/(length(cattr)+1);
		} else {
			probs <- append(c(1-sum(probs)), probs);
		}
		
		repl <- apply(as.array(1:nrow(data)), 1, 
			function(i) {
				if(length(which(data[i,cattr]>0))>1) {
					ret <- numeric(length(cattr));
					idx <- sample.int(length(cattr)+1, 1, prob=probs);
					if(idx > 0) ret[idx-1] <- trueval;
					
					return(ret);
				} else {
					return(as.numeric(data[i,cattr]));
				}
			});
		repl <- t(repl);
		data[,cattr] <- repl;
	}
	
	return(data);
}

fixupVectorAttributesPrivate <- function(data, attrs, trueval) {
	
	if(length(attrs) <= 0) return(data);
	
	n <- nrow(data);
	
	for(i in 1:length(attrs)) {		
		
		repl <- apply(as.array(1:nrow(data)), 1, 
			function(i) {
				if(length(which(data[i,cattr]>0))>1) {
					ret <- numeric(length(cattr));
					
					return(ret);
				} else {
					return(as.numeric(data[i,cattr]));
				}
			});
		repl <- t(repl);
		data[,cattr] <- repl;
	}
	
	return(data);
}

Mode <- function(x) {
  ux <- unique(x);
  ux[which.max(tabulate(match(x, ux)))]
};

vectorAttrMode <- function(data, guess) {
	
	means <- numeric(0);
	
	means <- append(means, (nrow(data) - sum(data[,guess]))/nrow(data));
	
	for(attr in guess) {
		means <- append(means, mean(data[,attr]));
	}
	
	return(which.max(means)-1);
};

vectorAttrProbs <- function(data, guess) {
	
	means <- numeric(0);
	
	means <- append(means, (nrow(data) - sum(data[,guess]))/nrow(data));
	
	for(attr in guess) {
		means <- append(means, mean(data[,attr]));
	}
	
	return(means);
};

getVectorAttrVal <- function(sample, guess) {
	whichguess <- which(sample[guess] > 0);
	if(length(whichguess) == 0) {
		return(0);
	} else {
		return(whichguess)
	}
	
	#if(sum(sample[guess]) == 0) {
	#	return(0);
	#} else {
	#	return(which(sample[guess] > 0));
	#}
};

isVectorWildType <- function(sample, guess) {
	if(sum(sample[guess]) == 0) {
		return(1);
	} else {
		return(0);
	}
};

row.matches <- function(y, X) {
 	i <- seq(nrow(X))
 	j <- 0
 	while(length(i) && (j <- j + 1) <= ncol(X)) 
 		i <- i[X[i, j] == y[j]]
 	i
}

getEntropyAnonDataset <- function(data, validation, missing, guess, ncut) {
	
	dodisc <- ncut > 0;	
	
	if(dodisc) {
		catted <- rbind(data, validation);
		dcatted <- round(numericFromNominal(discretizeDataset(catted, ncut)));
			
		data <- dcatted[1:nrow(data),];
		validation <- dcatted[(nrow(data)+1):(nrow(data)+nrow(validation)),];		
	} else {
		data <- round(data);
		validation <- round(validation);			
	}	
	
	dnames <- dimnames(validation)[[2]];
	names(data) <- dnames;
	known <- dnames[!(dnames %in% missing)];
	data <- data[,known];	
	
	baseentropy <- entropy(numeric(length(guess)+1)+1/(length(guess)+1));
	entropies <- numeric(0);
	
	for(i in 1:nrow(validation)) {
		
		sample <- validation[i,known];
		
		fmlas <- apply(as.array(known[!(known %in% guess)]), 1, function(name) sprintf("data[,%d] == %f", match(name, known), sample[match(name, known)]));
		selector <- paste(fmlas, collapse=" & ");
		
		ntrain <- data[which(eval(parse(text=selector))),];
		
		if(nrow(ntrain) == 0) {
			entropies <- append(entropies, baseentropy);
		} else {
			probs <- 1 - (sum(ntrain[,guess] > 0))/nrow(ntrain);
			probs <- append(probs, apply(as.array(guess), 1, function(name) sum(ntrain[,name] > 0)/nrow(ntrain)));
			#cat(probs, entropy(probs), "\n");
			
			entropies <- append(entropies, entropy(probs));
		}
		
		#cat(mean(entropies), " ");
	}
	
	return(mean(entropies));
}

getEntropyAnonDatasets <- function(fileprefix, nsets, validation, ncut, missing, guess) {

	validation <- as.matrix(validation);
	
	res <- numeric(0);
	
	for(i in 1:nsets) {
		
		filename <- paste0(fileprefix, "-", as.character(i), ".csv");
		
		#print(sprintf("loading %s", filename));
		
		data <- read.csv(filename);
		names(data) <- dimnames(validation)[[2]];
		data <- as.matrix(data);
				
		res <- append(res, getEntropyAnonDataset(data, validation, missing, guess, ncut));
		print(mean(res));
	}
	cat("\n");
	
	return(mean(res));	
}

getFullMemoryClassifier <- function(training, known, guess, mode) {
	
	unqdata <- lapply(known, function(name) unique(training[,name]));
	names(unqdata) <- known;
	knowncombs <- unique(fixupVectorAttributes(expand.grid(unqdata), getVectorAttributes(known), 1));
	row.names(knowncombs) <- 1:nrow(knowncombs);
	
	cmatrix <- matrix(data=0, nrow=nrow(knowncombs), ncol=length(known), byrow=TRUE);
	colnames(cmatrix) <- known;
	answers <- numeric(nrow(knowncombs));
	
	for(i in 1:nrow(knowncombs)) {
		
		sample <- knowncombs[i,];
		
		dnames <- dimnames(training)[[2]];
		fmlas <- apply(as.array(known), 1, function(name) sprintf("training[,%d] == %f", match(name, dnames), sample[match(name, known)]));
		selector <- paste(fmlas, collapse=" & ");
		
		ntrain <- training[which(eval(parse(text=selector))),];

		if(!is.null(nrow(ntrain)) && nrow(ntrain) > 0) {
			answer <- vectorAttrMode(ntrain, guess);
		} else {
			answer <- mode;
		}
		
		cmatrix[i,1:length(known)] <- as.numeric(sample);
		answers[i] <- as.numeric(answer);
	}
	
	return(function(sample) {
		
		matches <- row.matches(as.numeric(sample[known]), cmatrix);
		if(length(matches) <= 0) return(list(guess=mode, real=getVectorAttrVal(sample, guess)));
		return(list(guess=answers[matches[1]], real=getVectorAttrVal(sample, guess)));
	});
}

guessTargetAnonDataset <- function(training, sample, known, guess, mode) {	
	
	dnames <- dimnames(training)[[2]];
	fmlas <- apply(as.array(known), 1, function(name) sprintf("training[,%d] == %f", match(name, dnames), sample[match(name, dnames)]));
	selector <- paste(fmlas, collapse=" & ");
	
	training <- training[which(eval(parse(text=selector))),];
	if(!is.null(nrow(training)) && nrow(training) > 0) {
		return(list(guess=vectorAttrMode(training, guess), real=getVectorAttrVal(sample, guess)));
	} else {
		return(list(guess=mode, real=getVectorAttrVal(sample, guess)));
	}
}

normalizeDiscretization <- function(training, validation, ncut, dodisc) {
	
	if(dodisc) {
		catted <- rbind(training, validation);
		dcatted <- round(numericFromNominal(discretizeDataset(catted, ncut)));
			
		training <- dcatted[1:nrow(training),];
		validation <- dcatted[(nrow(training)+1):(nrow(training)+nrow(validation)),];
	} else {
		training <- round(training);
		validation <- round(validation);		
	}

	return(list(training=training, validation=validation));
}

getAnonDatasetClassifier <- function(training, ncut, guess, missing, dodisc) {
	
	anames <- dimnames(training)[[2]];
	known <- anames[!(anames %in% missing)];
	known <- known[!(known %in% guess)];		
	
	mode <- vectorAttrMode(training, guess);
	
	cls <- NA;
	if(dodisc) {
		cls <- getFullMemoryClassifier(training, known, guess, mode);
	} else {		
		cls <- function(sample) guessTargetAnonDataset(training, sample, known, guess, mode);
	}		

}

runAnonDatasetTrials <- function(validation, guess, cls, n=nrow(validation)) {	
		
	response <- numeric(0);
	observation <- numeric(0);	
	
	nguesstypes <- length(guess);	
	errortypes <- matrix(0, nrow = nguesstypes+1, ncol = nguesstypes+1);
	
	for(i in 1:n) {
		
		sample <- validation[i,];
		
		result <- cls(sample);
		
		response <- append(response, result$real);
		observation <- append(observation, result$guess);
		
		errortypes[result$real+1, result$guess+1] <- errortypes[result$real+1, result$guess+1] + 1;
		
		if(i %% 1000 == 0) {
			print(sprintf("acc: %f, fmeas: %f, auc: %f", sum(diag(errortypes))/i, fmeasurevector(errortypes, validation[1:i,], guess), as.numeric(multiclass.roc(response=response, predictor=observation)$auc)));
		}
	}		
	
	return(list(confusion = errortypes, accuracy = sum(diag(errortypes))/n, fmeas = fmeasurevector(errortypes, validation, guess), auc = as.numeric(multiclass.roc(response=response, predictor=observation)$auc)));
}

runAnonDatasetTrials.wildtype <- function(validation, guess, cls, n=nrow(validation)) {	
		
	response <- numeric(0);
	observation <- numeric(0);	
	
	nguesstypes <- length(guess);	
	errortypes <- matrix(0, nrow = 2, ncol = 2);
	
	mode <- vectorAttrMode(training, guess);
	
	for(i in 1:n) {
		
		sample <- validation[i,];
		
		result <- cls(sample);
		
		response <- append(response, as.numeric(result$real==mode)+1);
		observation <- append(observation, as.numeric(result$guess==mode)+1);
		
		errortypes[as.numeric(result$real==mode)+1, as.numeric(result$guess==mode)+1] <- errortypes[as.numeric(result$real==mode)+1, as.numeric(result$guess==mode)+1] + 1;
		
		if(i %% 1000 == 0) {
			print(sprintf("acc: %f, fmeas: %f, auc: %f", sum(diag(errortypes))/i, fmeasurevector(errortypes, validation[1:i,], guess), as.numeric(multiclass.roc(response=response, predictor=observation)$auc)));
		}
	}		
	
	return(list(confusion = errortypes, accuracy = sum(diag(errortypes))/n, fmeas = fmeasurevector(errortypes, validation, guess), auc = as.numeric(multiclass.roc(response=response, predictor=observation)$auc)));
}

runAllAnonDatasetTrials <- function(fileprefix, nsets, validation, ncut, missing, guess) {

	accs <- numeric(0);
	aucs <- numeric(0);
	fscs <- numeric(0);	
	
	validation <- as.matrix(validation);	
	
	for(i in 1:nsets) {
		
		filename <- paste0(fileprefix, "-", as.character(i), ".csv");
		
		print(sprintf("loading %s", filename));
		
		data <- read.csv(filename);
		names(data) <- dimnames(validation)[[2]];
		data <- as.matrix(data);
		
		newdisc <- normalizeDiscretization(data, validation, ncut, TRUE);
		data <- newdisc$training;
		validation <- newdisc$validation;
		cls <- getAnonDatasetClassifier(data, ncut, guess, missing, TRUE);
		results <- runAnonDatasetTrials(validation, guess, cls);
		
		accs <- append(accs, results$accuracy);
		aucs <- append(aucs, results$auc);
		fscs <- append(fscs, results$fmeas);		
		
		print(sprintf("----acc = %f, fmeas = %f, auc = %f", mean(accs), mean(fscs), mean(aucs)));
	}
	
	return(list(accuracy = mean(accs), fmeas = mean(fscs), auc = mean(aucs)));
}

runAllAnonDatasetComparisons <- function(fileprefix, eps, nsets, training, validation, ncut, missing, guess, evalfun) {

	accs <- numeric(0);
	auc <- numeric(0);
	vaccs <- numeric(0);
	vauc <- numeric(0);

	newdisc <- normalizeDiscretization(training, validation, ncut, FALSE);
	data <- newdisc$training;
	valid <- newdisc$validation;
	cls <- getAnonDatasetClassifier(data, ncut, guess, missing, FALSE);
	results <- evalfun(data, guess, cls);
	vresults <- evalfun(valid, guess, cls);
	baselineacc <- baseline.acc(training, guess);
	baselineauc <- baseline.auc(training, guess);
	bdiffacc <- mean(results$accuracy)-mean(vresults$accuracy);
	bdiffauc <- results$auc-vresults$auc;	
	
	validation <- as.matrix(validation);	
	training <- as.matrix(training);	
	
	print(sprintf("blineacc: %f, blineauc: %f, bdiffacc: %f (%f - %f), bdiffauc: %f (%f - %f)", baselineacc, baselineauc, bdiffacc, mean(results$accuracy), mean(vresults$accuracy), bdiffauc, results$auc, vresults$auc));
	
	for(i in 1:nsets) {
		
		filename <- sprintf("%s%.2f.%d-%d.csv", fileprefix, eps, ncut, i);

		print(sprintf("runAllAnonDatasetComparisons-eps.%.2f-ncut.%d-missing.%d-guess.%s", eps, ncut, length(missing), strsplit(guess[1],"=")[[1]][1]));		
		print(sprintf("    loading %s", filename));
		
		data <- read.csv(filename);
		names(data) <- dimnames(validation)[[2]];
		data <- as.matrix(data);

		newdisc <- normalizeDiscretization(data, rbind(training,validation), ncut, TRUE);
		data <- as.matrix(newdisc$training);
		train <- as.matrix(newdisc$validation[1:nrow(training),]);
		valid <- as.matrix(newdisc$validation[(nrow(training)+1):(nrow(training)+nrow(validation)),]);
		cls <- getAnonDatasetClassifier(data, ncut, guess, missing, TRUE);
		nresults <- evalfun(train, guess, cls);
		nvresults <- evalfun(valid, guess, cls);
		
		accs <- append(accs, nresults$accuracy);
		auc <- append(auc, nresults$auc);
		vaccs <- append(vaccs, nvresults$accuracy);
		vauc <- append(vauc, nvresults$auc);
		
		if(i > 1) {

			accstats <- getTestStatistics(accs, vaccs, baselineacc, bdiffacc);
			aucstats <- getTestStatistics(auc, vauc, baselineauc, bdiffauc);

			print(sprintf("i = %d, train: (acc: %f, auc: %f) valid: (acc: %f, auc: %f) diff: (acc: %f, auc: %f)", i, mean(accs), mean(auc), mean(vaccs), mean(vauc), mean(accs)-mean(vaccs), mean(auc)-mean(vauc)));
			print(sprintf("  acc p: %s", accstats$summ));
			print(sprintf("  auc p: %s", aucstats$summ));
		} else {
			print(sprintf("train: (acc: %f p = %f, auc: %f p = %f) valid: (acc: %f, auc: %f)", mean(accs), Inf, mean(auc), Inf, mean(vaccs), mean(vauc)));
		}							
	}
	
	summ <- list();
	if(nsets > 1) {
		
		accstats <- getTestStatistics(accs, vaccs, baselineacc, bdiffacc);
		aucstats <- getTestStatistics(auc, vauc, baselineauc, bdiffauc);
		
		summ$base <- sprintf("train: (acc: %f, auc: %f) valid: (acc: %f, auc: %f) diff: (acc: %f, auc: %f)", mean(accs), mean(auc), mean(vaccs), mean(vauc), mean(accs)-mean(vaccs), mean(auc)-mean(vauc));
		summ$accp <- sprintf("  acc p: %s", accstats$summ);
		summ$aucp <- sprintf("  auc p: %s", aucstats$summ);
	} else {
		summ <- sprintf("train: (acc: %f p = %f, auc: %f p = %f) valid: (acc: %f, auc: %f)", mean(accs), Inf, mean(auc), Inf, mean(vaccs), mean(vauc));
	}
	
	curdir <- getwd();
	dir.create(file.path(curdir, "privacy-results"), showWarnings = FALSE);
	setwd(file.path(curdir, "privacy-results"));
	filename <- sprintf("runAllAnonDatasetComparisons-prefix.%s-eps.%.2f-ncut.%d-missing.%d-guess.%s-evalfun.%s-%s.Robject", basename(fileprefix), eps, ncut, length(missing), strsplit(guess[1],"=")[[1]][1], deparse(substitute(evalfun)), format(Sys.time(), "%b%d.%H.%M.%S"));
	dput(list(summary=summ, baseline = sprintf("blineacc: %f, blineauc: %f, bdiffacc: %f (%f - %f), bdiffauc: %f (%f - %f)", baselineacc, baselineauc, bdiffacc, mean(results$accuracy), mean(vresults$accuracy), bdiffauc, results$auc, vresults$auc), accs=accs, auc=auc, vaccs=vaccs, vauc=vauc), file=filename);
	setwd(curdir);
	
	return(list(summary=summ, baseline = sprintf("blineacc: %f, blineauc: %f, bdiffacc: %f (%f - %f), bdiffauc: %f (%f - %f)", baselineacc, baselineauc, bdiffacc, mean(results$accuracy), mean(vresults$accuracy), bdiffauc, results$auc, vresults$auc), accs=accs, auc=auc, vaccs=vaccs, vauc=vauc));
}

runAnonDatasetVkoWithDemo <- function(prefix) {
	for(eps in c(0.25, 1, 5, 20, 100)) {
		runAllAnonDatasetComparisons(prefix, eps, 100, training, validation, 2, c("amiodarone", "decr", "cyp2c9=13", "cyp2c9=12", "cyp2c9=23", "cyp2c9=33", "cyp2c9=22"), c("vkorc1=CT", "vkorc1=TT"), runAnonDatasetTrials);
		runAllAnonDatasetComparisons(prefix, eps, 100, training, validation, 2, c("amiodarone", "decr", "cyp2c9=13", "cyp2c9=12", "cyp2c9=23", "cyp2c9=33", "cyp2c9=22"), c("vkorc1=CT", "vkorc1=TT"), runAnonDatasetTrials.wildtype);
		
	}
}

runAnonDatasetVkoWithAll <- function(prefix) {
	for(eps in c(0.25, 1, 5, 20, 100)) {
		runAllAnonDatasetComparisons(prefix, eps, 100, training, validation, 2, c(), c("vkorc1=CT", "vkorc1=TT"), runAnonDatasetTrials);
		runAllAnonDatasetComparisons(prefix, eps, 100, training, validation, 2, c(), c("vkorc1=CT", "vkorc1=TT"), runAnonDatasetTrials.wildtype);
	}
}

runAnonDatasetCypWithDemo <- function(prefix) {
	for(eps in c(0.25, 1, 5, 20, 100)) {
		runAllAnonDatasetComparisons(prefix, eps, 100, training, validation, 2, c("amiodarone", "decr", "vkorc1=CT", "vkorc1=TT"), c("cyp2c9=13", "cyp2c9=12", "cyp2c9=23", "cyp2c9=33", "cyp2c9=22"), runAnonDatasetTrials);
		runAllAnonDatasetComparisons(prefix, eps, 100, training, validation, 2, c("amiodarone", "decr", "vkorc1=CT", "vkorc1=TT"), c("cyp2c9=13", "cyp2c9=12", "cyp2c9=23", "cyp2c9=33", "cyp2c9=22"), runAnonDatasetTrials.wildtype);
	}
}

runAnonDatasetCypWithAll <- function(prefix) {
	for(eps in c(0.25, 1, 5, 20, 100)) {
		runAllAnonDatasetComparisons(prefix, eps, 100, training, validation, 2, c(), c("cyp2c9=13", "cyp2c9=12", "cyp2c9=23", "cyp2c9=33", "cyp2c9=22"), runAnonDatasetTrials);
		runAllAnonDatasetComparisons(prefix, eps, 100, training, validation, 2, c(), c("cyp2c9=13", "cyp2c9=12", "cyp2c9=23", "cyp2c9=33", "cyp2c9=22"), runAnonDatasetTrials.wildtype);
	}
}


runManyRegress <- function(training, validation, class, eps, n) {

	accs <- numeric(0);

	training <- as.matrix(training);	
	validation <- as.matrix(validation);
	
	set.seed(1);
	
	for(i in 1:n) {
				
		clfun <- diffpLR(training, class, eps);
		results <- regstats(function(x) WarfDose(clfun(x)), validation, class);
		
		accs <- append(accs, results$relative);
		
		cat(sprintf("%f ", mean(accs)));
	}
	cat("\n");
	
	return(list(accuracy = mean(accs)));
}


runAnonClassifierTrials <- function(training, validation, missing, guess) {
	
	nguesstypes <- length(guess);	
	errortypes <- matrix(0, nrow = nguesstypes+1, ncol = nguesstypes+1);
	
	#if(nrow(training) > 5000) training <- training[sample.int(nrow(training), 5000),] 
	training[,missing] <- 0;
	validation[,missing] <- 0;
	
	clfun <- Logistic(as.formula(sprintf("%s ~ .", guess)), training);
	
	stats <- evaluate_Weka_classifier(clfun, newdata=validation, class=TRUE);
	
	return(list(accuracy = as.numeric(stats$details["pctCorrect"]), fmeas = mean(as.numeric(stats$detailsClass[,"fMeasure"])), auc = mean(as.numeric(stats$detailsClass[,"areaUnderROC"]))));	
}

runAllAnonClassifierTrials <- function(fileprefix, nsets, validation, missing, guess) {
	
	accs <- numeric(0);
	aucs <- numeric(0);
	fscs <- numeric(0);
	
	validation <- as.matrix(validation);
	nominalValid <- nominalGenes(validation);	
	
	for(i in 1:nsets) {
		
		filename <- paste0(fileprefix, "-", as.character(i), ".csv");
		
		print(sprintf("loading %s", filename));
		
		data <- read.csv(filename);
		names(data) <- dimnames(validation)[[2]];		
		data <- as.matrix(data);
		data <- nominalGenes(data);
		
		results <- runAnonClassifierTrials(data, nominalValid, missing, guess);
		
		aucs <- append(aucs, results$auc);
		fscs <- append(fscs, results$fmeas);
		accs <- append(accs, results$accuracy);
		
		print(sprintf("----acc = %f, fmeas = %f, auc = %f", mean(accs), mean(fscs), mean(aucs)));		
	}
	
	return(list(accuracy = mean(accs), fmeas = mean(fscs), auc = mean(aucs)));
}

runMultipleParameterTrials <- function(traindata, testdata, missing, class, guess, params, ntrials, usemode) {
	
	#set.seed(1);
	
	#missing <- match(missing, names(data));
	
	paramvals <- numeric(0);
	aucvals <- numeric(0);
	fvals <- numeric(0);
	accvals <- numeric(0);
	
	nsamples <- 1;
	
	for(eps in params) {
	
		aucs <- numeric(0);
		fmeas <- numeric(0);
		accs <- numeric(0);
	
		for(i in 1:(ifelse(length(missing) <= 0 && eps <= 0, 1, ntrials))) {
			
			print(sprintf("epsilon = %f, trial = %d", eps, i));
			
			#clfun <- diffpLR(traindata, class, eps);
			
			results <- guessTargetsVectorAttr(testdata, eps, missing, class, guess, nrow(testdata), ifelse(length(missing)<=0 || usemode, 1, nsamples), usemode);			
			
			aucs <- append(aucs, results$auc);
			fmeas <- append(fmeas, results$fmeas);
			accs <- append(accs, results$accuracy);
		}
		
		paramvals <- append(paramvals, eps);
		aucvals <- append(aucvals, mean(aucs));
		fvals <- append(fvals, mean(fmeas));
		accvals <- append(accvals, mean(accs));
		
		print(cbind(param = paramvals, auc = aucvals, fscore = fvals, acc = accvals));
	}
	
	return(cbind(param = paramvals, auc = aucvals, fscore = fvals, acc = accvals));
}

runKFoldTrials <- function(data, missing, class, guess, params, k, nruns, dononprivate) {
	
	set.seed(1);
	
	missing <- match(missing, names(data));
	
	nsamples <- 100;
	
	folds <- genfolds(data, k);
	
	paramvals <- numeric(0);
	ptestaucvals <- numeric(0);
	ptestfvals <- numeric(0);
	ptestaccvals <- numeric(0);
	nptestaucvals <- numeric(0);
	nptestfvals <- numeric(0);
	nptestaccvals <- numeric(0);
	mtestaucvals <- numeric(0);
	mtestfvals <- numeric(0);
	mtestaccvals <- numeric(0);
	ptrainaucvals <- numeric(0);
	ptrainfvals <- numeric(0);
	ptrainaccvals <- numeric(0);
	nptrainaucvals <- numeric(0);
	nptrainfvals <- numeric(0);
	nptrainaccvals <- numeric(0);
	mtrainaucvals <- numeric(0);
	mtrainfvals <- numeric(0);
	mtrainaccvals <- numeric(0);	

	nptestaucs <- numeric(0);
	nptestfmeas <- numeric(0);
	nptestaccs <- numeric(0);
	mtestaucs <- numeric(0);
	mtestfmeas <- numeric(0);
	mtestaccs <- numeric(0);
	nptrainaucs <- numeric(0);
	nptrainfmeas <- numeric(0);
	nptrainaccs <- numeric(0);
	mtrainaucs <- numeric(0);
	mtrainfmeas <- numeric(0);
	mtrainaccs <- numeric(0);	

	if(dononprivate) {	
		for(i in 1:k) {
		
			npclfun <- diffpLR(folds[[i]]$train, class, 0);		
		
			print(sprintf("mode test, fold = %d", i));
			mtestresults <- guessTargetsVectorAttr(folds[[i]]$test, NULL, missing, class, guess, nrow(folds[[i]]$test), 1, TRUE);
			print(sprintf("mode train, fold = %d", i));
			mtrainresults <- guessTargetsVectorAttr(folds[[i]]$train, NULL, missing, class, guess, nrow(folds[[i]]$train), 1, TRUE);
			
			print(sprintf("non-private test, fold = %d", i));
			nptestresults <- guessTargetsVectorAttr(folds[[i]]$test, npclfun, missing, class, guess, nrow(folds[[i]]$test), nsamples, FALSE);
			print(sprintf("non-private train, fold = %d", i));
			nptrainresults <- guessTargetsVectorAttr(folds[[i]]$train, npclfun, missing, class, guess, nrow(folds[[i]]$train), nsamples, FALSE);			
	
			nptestaucs <- append(nptestaucs, nptestresults$auc);
			nptestfmeas <- append(nptestfmeas, nptestresults$fmeas);
			nptestaccs <- append(nptestaccs, nptestresults$accuracy);
			mtestaucs <- append(mtestaucs, mtestresults$auc);
			mtestfmeas <- append(mtestfmeas, mtestresults$fmeas);
			mtestaccs <- append(mtestaccs, mtestresults$accuracy);
			nptrainaucs <- append(nptrainaucs, nptrainresults$auc);
			nptrainfmeas <- append(nptrainfmeas, nptrainresults$fmeas);
			nptrainaccs <- append(nptrainaccs, nptrainresults$accuracy);
			mtrainaucs <- append(mtrainaucs, mtrainresults$auc);
			mtrainfmeas <- append(mtrainfmeas, mtrainresults$fmeas);
			mtrainaccs <- append(mtrainaccs, mtrainresults$accuracy);			
	
			print(cbind(nptestaucs = mean(nptestaucs), nptestfmeas = mean(nptestfmeas), nptestaccs = mean(nptestaccs), nptrainaucs = mean(nptrainaucs), nptrainfmeas = mean(nptrainfmeas), nptrainaccs = mean(nptrainaccs), mtestaucs = mean(mtestaucs), mtestfmeas = mean(mtestfmeas), mtestaccs = mean(mtestaccs), mtrainaucs = mean(mtrainaucs), mtrainfmeas = mean(mtrainfmeas), mtrainaccs = mean(mtrainaccs)));
			cat("-------------------\n\n");
		}
	}
	
	for(eps in params) {
	
		ptestaucs <- numeric(0);
		ptestfmeas <- numeric(0);
		ptestaccs <- numeric(0);
		ptrainaucs <- numeric(0);
		ptrainfmeas <- numeric(0);
		ptrainaccs <- numeric(0);
			
		for(i in 1:k) {
			
			for(j in 1:nruns) {
				clfun <- diffpLR(folds[[i]]$train, class, eps);
				
				print(sprintf("private test, epsilon = %f, fold = %d", eps, i));
				ptestresults <- guessTargetsVectorAttr(folds[[i]]$test, clfun, missing, class, guess, nrow(folds[[i]]$test), nsamples, FALSE);
				print(sprintf("private train, epsilon = %f, fold = %d", eps, i));
				ptrainresults <- guessTargetsVectorAttr(folds[[i]]$train, clfun, missing, class, guess, nrow(folds[[i]]$train), nsamples, FALSE);
							
				ptestaucs <- append(ptestaucs, ptestresults$auc);
				ptestfmeas <- append(ptestfmeas, ptestresults$fmeas);
				ptestaccs <- append(ptestaccs, ptestresults$accuracy);
				ptrainaucs <- append(ptrainaucs, ptrainresults$auc);
				ptrainfmeas <- append(ptrainfmeas, ptrainresults$fmeas);
				ptrainaccs <- append(ptrainaccs, ptrainresults$accuracy);
	
				tparamvals <- append(paramvals, eps);
				tptestaucvals <- append(ptestaucvals, mean(ptestaucs));
				tptestfvals <- append(ptestfvals, mean(ptestfmeas));
				tptestaccvals <- append(ptestaccvals, mean(ptestaccs));
				tnptestaucvals <- append(nptestaucvals, mean(nptestaucs));
				tnptestfvals <- append(nptestfvals, mean(nptestfmeas));
				tnptestaccvals <- append(nptestaccvals, mean(nptestaccs));
				tmtestaucvals <- append(mtestaucvals, mean(mtestaucs));
				tmtestfvals <- append(mtestfvals, mean(mtestfmeas));
				tmtestaccvals <- append(mtestaccvals, mean(mtestaccs));
				tptrainaucvals <- append(ptrainaucvals, mean(ptrainaucs));
				tptrainfvals <- append(ptrainfvals, mean(ptrainfmeas));
				tptrainaccvals <- append(ptrainaccvals, mean(ptrainaccs));
				tnptrainaucvals <- append(nptrainaucvals, mean(nptrainaucs));
				tnptrainfvals <- append(nptrainfvals, mean(nptrainfmeas));
				tnptrainaccvals <- append(nptrainaccvals, mean(nptrainaccs));
				tmtrainaucvals <- append(mtrainaucvals, mean(mtrainaucs));
				tmtrainfvals <- append(mtrainfvals, mean(mtrainfmeas));
				tmtrainaccvals <- append(mtrainaccvals, mean(mtrainaccs));
				
				privateres <- cbind(param = tparamvals, testauc = tptestaucvals, testfscore = tptestfvals, testacc = tptestaccvals, trainauc = tptrainaucvals, trainfscore = tptrainfvals, trainacc = tptrainaccvals);
				nonprivateres <- cbind(param = tparamvals, testauc = tnptestaucvals, testfscore = tnptestfvals, testacc = tnptestaccvals, trainauc = tnptrainaucvals, trainfscore = tnptrainfvals, trainacc = tnptrainaccvals);
				moderes <- cbind(param = tparamvals, testauc = tmtestaucvals, testfscore = tmtestfvals, testacc = tmtestaccvals, trainauc = tmtrainaucvals, trainfscore = tmtrainfvals, trainacc = tmtrainaccvals);
				
				print(list(private = privateres, nonprivate = nonprivateres, mode = moderes));
				cat("-------------------\n\n");
			}
			
		}
		
		paramvals <- append(paramvals, eps);
		ptestaucvals <- append(ptestaucvals, mean(ptestaucs));
		ptestfvals <- append(ptestfvals, mean(ptestfmeas));
		ptestaccvals <- append(ptestaccvals, mean(ptestaccs));
		nptestaucvals <- append(nptestaucvals, mean(nptestaucs));
		nptestfvals <- append(nptestfvals, mean(nptestfmeas));
		nptestaccvals <- append(nptestaccvals, mean(nptestaccs));
		mtestaucvals <- append(mtestaucvals, mean(mtestaucs));
		mtestfvals <- append(mtestfvals, mean(mtestfmeas));
		mtestaccvals <- append(mtestaccvals, mean(mtestaccs));
		ptrainaucvals <- append(ptrainaucvals, mean(ptrainaucs));
		ptrainfvals <- append(ptrainfvals, mean(ptrainfmeas));
		ptrainaccvals <- append(ptrainaccvals, mean(ptrainaccs));
		nptrainaucvals <- append(nptrainaucvals, mean(nptrainaucs));
		nptrainfvals <- append(nptrainfvals, mean(nptrainfmeas));
		nptrainaccvals <- append(nptrainaccvals, mean(nptrainaccs));
		mtrainaucvals <- append(mtrainaucvals, mean(mtrainaucs));
		mtrainfvals <- append(mtrainfvals, mean(mtrainfmeas));
		mtrainaccvals <- append(mtrainaccvals, mean(mtrainaccs));		
	}

	privateres <- cbind(param = paramvals, testauc = ptestaucvals, testfscore = ptestfvals, testacc = ptestaccvals, trainauc = ptrainaucvals, trainfscore = ptrainfvals, trainacc = ptrainaccvals);
	nonprivateres <- cbind(param = paramvals, testauc = nptestaucvals, testfscore = nptestfvals, testacc = nptestaccvals, trainauc = nptrainaucvals, trainfscore = nptrainfvals, trainacc = nptrainaccvals);
	moderes <- cbind(param = paramvals, testauc = mtestaucvals, testfscore = mtestfvals, testacc = mtestaccvals, trainauc = mtrainaucvals, trainfscore = mtrainfvals, trainacc = mtrainaccvals);
	
	return(list(private = privateres, nonprivate = nonprivateres, mode = moderes));	
};

guessTargets <- function(data, model, confusion, missing, class, guess, n, isrounded, epsilon, nclassifiers, usemode) {

	set.seed(1);
	guessmode <- Mode(data[,guess]);
	
	nguesstypes <- length(unique(data[,guess]));
	
	errortypes <- matrix(0, nrow = nguesstypes, ncol = nguesstypes);
	predictions <- c();

	avgcorrect <- 0;
	rarecorrect <- 0;
	t <- 0;
	numrare <- 0;
	
	#data <- data[data$vkorc1 != -1.0, ];
	#data <- transform(data, height = log(height), weight = log(weight), dose = round(dose));
	
	if(n > nrow(data)) n <- nrow(data);
	
	#maxsum <- 0;
	#for(i in 1:nrow(data)) {
	#	sum <- sqrt(sum(data[i,]^2));
	#	if(is.na(sum)) next;
	#	if(sum > maxsum) maxsum <- sum;
	#}
	
	maxsum <- 1;
	#data <- data*(1/maxsum);
	
	#scaling <- scaled(dose ~ ., data);
	#data <- data.frame(scaling[["data"]]);
	#maxsum <- scaling[["scale"]];

	for(i in 1:n) {
		
		value <- data[i,class];
		realattr <- data[i, guess];
		
		if(is.na(value) || is.na(realattr)) next;
		
		#guessval <- guessTargetBlackbox(data, confusion, model, value, missing, class, guess, i, isrounded, maxsum, usemode);
		guessval <- guessTargetOptim(data, model, missing, class, guess, i, isrounded, usemode)
				
		if(i %% 10 == 0) print(sprintf("avgcorrect = %f", (avgcorrect)/(t)));
		
		guessed <- as.numeric(guessval[1]);
		#guessed <- factor(guessval[1]);
		#levels(guessed) = levels(isrounded);
		
		errortypes[as.numeric(realattr), guessed] <- errortypes[as.numeric(realattr), guessed] + 1;
		predictions <- append(predictions, guessed);
		
		if(is.na(guessed)) next;
		
		#print(sprintf("realattr = %s, guessval = %s, eq: %s", realattr, isrounded[guessed], as.numeric(realattr) == guessed));
		
		if(as.numeric(realattr) != as.numeric(guessmode)) {
			
			numrare <- numrare + 1;
			
			if(as.numeric(realattr) == guessed) rarecorrect <- rarecorrect + 1;
			
			#if(numrare %% 100 == 0) print(sprintf("rare correct %%: %f", rarecorrect / numrare));
		}
		if(as.numeric(realattr) == guessed) avgcorrect <- avgcorrect + 1;
		t <- t + 1;
	}
	
	#print(sprintf("# actuals: %d, # predictions: %d", length(data[,guess]), length(predictions)));
	
	#print(errortypes);
	
	list(accuracy = avgcorrect / t, confusion = errortypes, fmeas = fmeasure(errortypes, data, guess), auc = aucroc(errortypes, data, guess));
};

guessTargetBlackbox <- function(data, confusion, model, value, missing, class, guess, idx, rounding, scalefactor, usemode) {
	
	attrs <- attributes(data)$names;	
	guesses <- c();
	printed <- FALSE;
	origval <- value;
	
	actual <- data[idx,guess];
	guessmode <- Mode(data[,guess]);
	
	if(usemode) return(guessmode);
	
	guesstabs <- tabulate(data[,guess]);
	
	sample <- data[idx,];
	
	#data <- as.matrix(data);
	#row.names(data) <- 1:nrow(data);
	
	#print(sprintf("mode: %s", guessmode));
	
	classlevels <- levels(data[,class]);
	
	#value <- predict(model, data[idx,]);
	#value <- value + rnorm(1, mean = mean(model$resid), sd = sd(model$resid));
	#value <- mean(warf[,class]);
	
	#print(sprintf("value: %f", value));

	ntrials <- 1;
	if(length(missing) > 0) ntrials <- 100;

	for(n in 1:ntrials) {
		
		#print(sprintf("real value: %s, sampled value: %s", origval, value));		
		
		if(length(missing) > 0) {
			
			#atridxs <- round(runif(length(missing), 1, nrow(data)));
			#sample[missing] <- data[cbind(atridxs, missing)];			
			
			for(i in 1:length(missing)) {
								
				attr <- missing[i];				
				atridx <- runif(1, 1, nrow(data));
				val <- data[atridx, attr];
				
				sample[attr] <- val;
			}
		}
				
		sample[guess] <- guessmode;
		
		curguess <- guessmode;
				
		if(is.numeric(value)) {
			
			mindist <- abs(predict(model, sample) - value);
			
			if(is.na(mindist)) next;
			
			for(i in 1:length(rounding)) {
				sample[guess] <- rounding[i];
				dist <- abs(predict(model, sample) - value);
				if(dist < mindist && dist < 0.25) {
					curguess <- rounding[i];
					mindist <- dist;
				}
			}
		} else if(!usemode) {
			
			ridxs <- seq(1, length(rounding))[sort.list(guesstabs + rnorm(length(guesstabs), 0, sqrt(var(guesstabs))), decreasing=TRUE)];
			#ridxs <- seq(1, length(rounding))[sort.list(guesstabs, decreasing=TRUE)];
			
			#print(sample);
			
			for(i in ridxs) {

				#sample[guess] <- rounding[i];
				#ridx <- runif(1, 1, nrow(data));
				#sample[guess] <- data[ridx, guess];
				#ridx <- runif(1, 1, length(rounding));
				sample[guess] <- factor(rounding[i], rounding); #rounding[i];
				#print(sample);
				pr <- NA;
				if(is.function(model)) {
					#print(sample);
					pr <- model(sample);
				} else {
					pr <- predict(model, sample, type="class");
				}

				#print(pr);
				#print(sample);
				#print(sprintf("    predicted = %s (value = %s)", pr, value));
				
				if(is.na(pr)) next;
				
				#rs <- runif(1, 0, 1);

				if(pr == value) {
				#if(rs <= confusion[value, pr]) {
					#print("    found match!")
					curguess <- sample[guess];
					break;
				}
			}
			
			#print(sprintf("guess: %s, actual: %s (predicted = %s, value = %s)", as.numeric(curguess), as.numeric(actual), pr, value));
		}
		
		printed <- TRUE;
		
		#print(sprintf("appending guess: %s (%s)", curguess, typeof(curguess)));
				
		guesses <- c(guesses, curguess); # append(guesses, curguess);
	}
		
	if(is.null(guesses)) {
		ret <- guessmode;
	} else {
		ret <- Mode(guesses);
	}
		
	c(ret, actual)
};

guessTargetOptim <- function(data, model, missing, class, guess, idx, rounding, usemode) {
	
	value <- data[idx, class];
	attrs <- attributes(data)$names;	
	guesses <- c();
	printed <- FALSE;
	origval <- value;
	nrows <- as.numeric(nrow(data));
	ncols <- as.numeric(ncol(data));

	actual <- data[idx,guess];
	guessmode <- Mode(data[,guess]);
	
	if(usemode) return(guessmode);				
	
	sample <- data[idx,];
	dlevels <- apply(as.array(attrs), 1, function(x) levels(data[,x]));
	names(dlevels) <- attrs;
	
	replace <- append(missing, guess);
	
	modes <- apply(as.array(missing), 1, function(x) {m <- Mode(data[,x]); ifelse(is.null(dlevels[[x]]), m, factor(dlevels[[x]][m], dlevels[[x]])) });
	minbounds <- apply(as.array(replace), 1, function(x) { ifelse(is.null(dlevels[[x]]), min(data[,x]), 1.0) });
	maxbounds <- apply(as.array(replace), 1, function(x) { ifelse(is.null(dlevels[[x]]), max(data[,x]), as.numeric(length(dlevels[[x]]))) });
	incrs <- apply(as.array(replace), 1, function(x) { ifelse(is.numeric(data[1,x]), (max(data[,x]) - min(data[,x]))/10, 1) });
	
	filtered <- dlevels[Filter(function(x) { if(x %in% replace) { return(TRUE); } else { return(FALSE); }}, names(dlevels))];
	
	data <- as.matrix(data);
	
	vals <- Map(
		function(name) {
			x <- dlevels[[name]]; 
			if(is.null(x)) { 
				mindata <- min(data[,name]); 
				maxdata <- max(data[,name]); 
				return(seq(mindata, maxdata, round((maxdata-mindata)/10)));
			} else {
				return(x);
			}
		}, names(filtered));
		
	grid <- expand.grid(vals);
	
	distrs <- apply(as.array(attrs), 1, 		
		function(x) { 			
			if(is.null(dlevels[[x]])) {
				meanx <- mean(as.numeric(data[,x]));
				sdx <- sd(as.numeric(data[,x]));
				function(v) {
					return(dnorm(as.numeric(v), mean=meanx, sd=sdx));
				}
			} else {
				function(v) {
					#print(sprintf("v: %s, x: %s", v, x));
					sumx <- sum(data[,x] == factor(dlevels[[x]][as.numeric(v)], dlevels[[x]]));
					#print(sumx);
				 	return(sumx/nrows);
				}
			}
		}
	);
	names(distrs) <- attrs;
	fnmaps <- apply(as.array(append(missing, guess)), 1, 
		function(x) { 
			if(is.null(dlevels[[x]])) {
				function (v) { return(v); }; 
			} else { 
				function(v) { return(round(as.numeric(v))); };
			}
		}
	);
	fnmap <- function(v) {
		rv <- apply(as.array(1:(length(v))), 1, function(i) fnmaps[[i]](v[i]));
		return(rv);
	}
				
	#print(sample);
	
	objective <- function(v) {
		
		modsample <- sample;
		#missingvals <- apply(as.array(1:(length(missing))), 1, 
		#	function (x) { 
		#		if(is.null(dlevels[[missing[x]]])) {
		#			return(v[x]);
		#		} else {
		#			#print(missing[x]);
		#			#print(sample[missing[x]]);
		#			#print(levels(sample[missing[x]]));
		#			return(factor(dlevels[[missing[x]]][v[x]], dlevels[[missing[x]]]));
		#		}
		#	}
		#);
		#modsample[guess] <- factor(rounding[v[length(missing)+1]], dlevels[[guess]]);
		
		for(n in names(v)) {
			modsample[n] <- factor(v[n], dlevels[[n]]);
		}
		#modsample[names(repl)] <- repl;
		#modsample[names(repl)[1]] <- repl[1];

		#print("----------------");
		#print(sample);
		#print("----------------");		
		#print(as.numeric(modsample));
		#print(model(modsample));
		#print(origval);
		#print("----------------");
		
		if(model(modsample) == origval) {

			#print(sprintf("guess: %s, real: %s", factor(rounding[v[length(missing)+1]], levels(data[,guess])), actual));
			rv <- ncols - sum(apply(as.array(replace), 1, function(name) (distrs[[name]])(modsample[name])));
			#rv <- rv + distrs[[guess]](modsample[guess]);
			#print(apply(as.array(1:(length(modsample))), 1, function(i) (distrs[[i]])(modsample[i])));

			return(rv);
		} else {

			return(ncols);
		}
	}
	
	optvals <- apply(grid, 1, objective);
	optq <- as.numeric(quantile(optvals, c(0.25)));
	optvals <- which(optvals <= optq);
	
	#print(optvals);
	#print(grid[optvals,guess]);
	
	guessed <- Mode(grid[optvals,guess]);
	print(sprintf("guessed: %s, actual: %s", guessed, actual));
	
	#optrsol <- DEoptim(objective, lower=minbounds, upper=maxbounds, control=DEoptim.control(itermax = 2, trace=FALSE), fnMap=fnmap)$optim$bestmem[length(missing)+1];
	#optrsol <- DEoptim(objective, lower=minbounds, upper=maxbounds);
	
	#optrsol <- factor(dlevels[[guess]][optrsol], dlevels[[guess]]);
	
	#list(guessed=optrsol, actual=actual);
	#return(c(optrsol, actual));
	return(c(guessed, actual));
};


guessTargetVectorAttr <- function(data, model, missing, class, guess, actual, mode, idx, ntrials, usemode) {
	
	lnames <- length(data);
	names <- dimnames(data)[[2]];
	missidxs <- NA;
	if(length(missing) > 0) missidxs <- apply(as.array(missing), 1, function(m) { which(names == m) });
	guessprobs <- vectorAttrProbs(data, guess);
	
	if(!usemode) data <- as.matrix(data);
	
	guesses <- numeric(length(guess)+1);
	classval <- data[idx,class];

	origsample <- data[idx,];

	hasval <- 1.0;
	
	for(n in 1:ntrials) {
		
		sample <- origsample;

		if(length(missing) > 0 && !usemode) {
			
			missvals <- apply(as.array(missing), 1, function(x) { idx <- sample.int(nrow(data), 1); data[idx, x] });
			sample[missing] <- missvals;

			#atridxs <- round(runif(1, 1, nrow(data))) + numeric(length(missing)); # round(runif(length(missing), 1, nrow(data)));			
			#sample[missing] <- data[cbind(atridxs, missidxs)];
		}		
	
		mindist <- 100;
		guessed <- mode;
		if(!usemode) {
			for(i in 0:length(guess)) {
				
				sample[guess] <- 0.0;
				if(i > 0) sample[guess[i]] <- hasval;
				
				pr <- NA;
				if(!is.function(model)) {
					pr <- predict(model, sample);
				} else {
					pr <- model(sample);
				}
				
				dist <- abs(pr - classval);
				guesses[i+1] <- dist;
			}
		}		
	}
	
	guessed <- which.min(guesses)-1;
	
	return(guessed);
};

guessTargetsVectorAttr <- function(data, model, missing, class, guess, n, nsamples, usemode) {

	#set.seed(1);
	
	numprocessed <- 0;
	nguesstypes <- length(guess);
	errortypes <- matrix(0, nrow = nguesstypes+1, ncol = nguesstypes+1);
	
	mode <- vectorAttrMode(data, guess);
	
	if(n > nrow(data)) n <- nrow(data);
	
	for(i in 1:n) {
		
		sample <- data[i,];
		value <- sample[class];
		realattr <- getVectorAttrVal(sample, guess);
		
		umodel <- NA;
		if(is.numeric(model)) {
			umodel <- diffpLR(data, class, model);
		} else {
			umodel <- model;
		}
				
		if(is.na(value) || is.na(realattr)) next;
		
		guessval <- guessTargetVectorAttr(data, umodel, missing, class, guess, realattr, mode, i, nsamples, usemode);
		
		errortypes[realattr+1, guessval+1] <- errortypes[realattr+1, guessval+1] + 1;
		
		numprocessed <- numprocessed + 1;
		
		if(numprocessed %% 100 == 0) print(sprintf("current accuracy (i = %d): %f", numprocessed, sum(diag(errortypes))/numprocessed));
	}
	
	cat("\n-------------------\n");

	return(list(confusion = errortypes, accuracy = sum(diag(errortypes))/numprocessed, fmeas = fmeasurevector(errortypes, data, guess), auc = aucrocvector(errortypes, data, guess)));
};