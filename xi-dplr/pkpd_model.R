# implementation of the PK-PD model from Hamberg et al., 2007
# for background on two-compartment models, I referenced documents at boomer.org (http://www.boomer.org/c/p4/c19/c19.pdf)
# as well as "Applied Clinical Pharmacokinetics", Bauer, McGraw-Hill 2008.


library(deSolve);
library(compiler);
library(msm);

# we only return doses to the nearest 1/4 milligram
doseincr <- 0.5;
RoundUp <- function(d, x) ifelse(d - x < 0, 16, abs(d - x));
RoundDown <- function(d, x) ifelse(d - x > 0, 16, abs(d - x));
WarfDoseUp <- function(x) ifelse(x >= 15, 15, seq(0,15,by=doseincr)[which.min(RoundUp(seq(0,15,by=doseincr), max(x,0.5)))]);
WarfDoseDown <- function(x) seq(0,15,by=doseincr)[which.min(RoundDown(seq(0,15,by=doseincr), max(x,0.5)))]; 
WarfDose <- function(x) seq(0,15,by=doseincr)[which.min(abs(seq(0,15,by=doseincr) - max(x,0.5)))];

strokes <- function(ttrs) {
	sum(apply(as.array(ttrs), 1, function(x) ifelse(x < 0.6, 2.1/length(ttrs), ifelse(x >= 0.6 && x <= 0.75, 1.34/length(ttrs), 1.07/length(ttrs)))));
}

bleeds <- function(ttrs) {
	sum(apply(as.array(ttrs), 1, function(x) ifelse(x < 0.6, 43.64/length(ttrs), ifelse(x >= 0.6 && x <= 0.75, 41.81/length(ttrs), 34.05/length(ttrs)))));
}

deaths <- function(ttrs) {
	sum(apply(as.array(ttrs), 1, function(x) ifelse(x < 0.6, 4.2/length(ttrs), ifelse(x >= 0.6 && x <= 0.75, 1.84/length(ttrs), 1.69/length(ttrs)))));
}

CoumagenRegression <- function(sample) {
	age <- as.numeric(25 + 10*(sample["age"]-1))*(-0.009);
	
	sex <- 0.0;
	if(runif(1, 0, 1) <= 0.51)
		sex <- 0.094;
		
	weight <- 0.454*sample["weight"]*0.003;
	
	cyp2c9 <- 0;
	if(sample["cyp2c9=12"] > 0) cyp2c9 <- -0.197;
	if(sample["cyp2c9=13"] > 0) cyp2c9 <- -0.360;
	if(sample["cyp2c9=22"] > 0) cyp2c9 <- -0.265;
	if(sample["cyp2c9=23"] > 0) cyp2c9 <- -0.947;
	if(sample["cyp2c9=33"] > 0) cyp2c9 <- -1.892;
	
	vkorc1 <- 0;
	if(sample["vkorc1=CT"] > 0) vkorc1 <- -0.304;
	if(sample["vkorc1=TT"] > 0) vkorc1 <- -0.569;
	if(sample["vkorc1=CC"] > 0) vkorc1 <- 0;	

	#print(sprintf("coumagen dose: %f", (1.64 + exp(3.984 + age + sex + weight + cyp2c9 + vkorc1))/7));
	return((1.64 + exp(3.984 + age + sex + weight + cyp2c9 + vkorc1))/7);
}

WilsonAdjuster <- function(time, inr, dose, lpdose, origdose) {

	if(time < 48) {
		
		return(list(dose=GetDoseFunc(time, c(dose*2, dose*2)), nextappt=time + 48))
	
	} else {	
		
		if(inr < 1.4) {
			return(list(dose=GetDoseFunc(time, numeric(5)+WarfDoseUp(dose+dose*0.5)), nextappt=time + 5*24));
		} else if(1.4 <= inr && inr < 1.5) {
			return(list(dose=GetDoseFunc(time, numeric(5)+WarfDoseUp(dose+dose*0.33)), nextappt=time + 5*24));
		} else if(1.5 <= inr && inr < 1.9) {
			return(list(dose=GetDoseFunc(time, numeric(5)+WarfDoseUp(dose+dose*0.25)), nextappt=time + 5*24));
		} else if(1.9 <= inr && inr < 2.0) {
			return(list(dose=GetDoseFunc(time, numeric(7)+WarfDoseUp(dose+dose*0.10)), nextappt=time + 7*24));
		} else if(2.0 <= inr && inr < 2.9) {
			return(list(dose=GetDoseFunc(time, numeric(14)+dose), nextappt=time + 14*24));
		} else if(2.9 <= inr && inr < 3.2) {
			return(list(dose=GetDoseFunc(time, numeric(7)+WarfDoseDown(dose-dose*0.10)), nextappt=time + 7*24));
		} else if(3.2 <= inr && inr < 3.6) {
			return(list(dose=GetDoseFunc(time, numeric(7)+WarfDoseDown(dose-dose*0.25)), nextappt=time + 7*24));
		} else if(3.6 <= inr && inr < 3.8) {
			return(list(dose=GetDoseFunc(time, numeric(7)+WarfDoseDown(dose-dose*0.33)), nextappt=time + 7*24));
		} else if(3.8 <= inr && inr < 4.0) {
			return(list(dose=GetDoseFunc(time, append(c(0), numeric(4)+WarfDoseDown(dose-dose*0.33))), nextappt=time + 5*24));
		} else if(4.0 <= inr && inr < 4.5) {
			return(list(dose=GetDoseFunc(time, append(c(0), numeric(2)+WarfDoseDown(dose-dose*0.33))), nextappt=time + 3*24));
		} else if(4.5 <= inr && inr < 5.1) {
			return(list(dose=GetDoseFunc(time, append(c(0, 0), numeric(1)+WarfDoseDown(dose-dose*0.33))), nextappt=time + 3*24));
		} else if(5.1 <= inr) {
			return(list(dose=GetDoseFunc(time, append(c(0, 0, 0), numeric(1)+WarfDoseDown(dose-dose*0.5))), nextappt=time + 4*24));
		}
		
	}
}

# this is the clinical dosing algorithm from the UMich hospital website,
# simplified so that there is no randomness in the next appointment time or dosage
AdjustDose <- function(time, inr, dose, lpdose, origdose) {
	
	if(time < 24) {
		return(list(dose=GetDoseFunc(time, numeric(7)+dose), nextappt=time + 7*24))
	}
	
	if(inr >= 0 && inr < 1.5)
		return(list(dose=GetDoseFunc(time, numeric(4)+dose+dose*0.2), nextappt=time + 4*24))
	else if(inr >= 1.5 && inr < 2.0)
		return(list(dose=GetDoseFunc(time, numeric(7)+dose+dose*0.1), nextappt=time + 7*24))
	else if(inr >= 2.0 && inr <= 3.0)
		return(list(dose=GetDoseFunc(time, numeric(7)+dose), nextappt=time + 7*24))
	else if(inr > 3.0 && inr < 4.0)
		return(list(dose=GetDoseFunc(time, numeric(7)+dose-dose*0.1), nextappt=time + 7*24))
	else if(inr >= 4.0)
		return(list(dose=GetDoseFunc(time, c(0, 0, dose-dose*0.2, dose-dose*0.2)), nextappt=time + 4*24))
};

AdjustDoseIdentity <- function(time, inr, dose, lpdose, origdose) {
	return(list(dose=function(x) return(origdose), nextappt=time+24*14));
}

GetDoseFunc <- function(time, doses) {

	function(t) {
		return(WarfDose(doses[(t-time)/24 + 1]));
	}	
}

AdjustDoseCoumagen <- function(time, inr, dose, lpdose, origdose) {	
	
	if(time < 48) {
		
		return(list(dose=GetDoseFunc(time, c(dose*2, dose*2)), nextappt=time + 48))
		
	} else if(time >= 48 && time < 4*24) {
				
		if(inr < 1.3)
			return(list(dose=GetDoseFunc(time, c(15, 15)), nextappt=time + 48))
		else if(1.3 <= inr && inr < 1.5)
			return(list(dose=GetDoseFunc(time, c(10, 10)), nextappt=time + 48))
		else if(1.5 <= inr && inr < 1.7)
			return(list(dose=GetDoseFunc(time, c(10, 5)), nextappt=time + 48))
		else if(1.7 <= inr && inr < 2.0)
			return(list(dose=GetDoseFunc(time, c(5, 5)), nextappt=time + 48))
		else if(2.0 <= inr && inr < 2.3)
			return(list(dose=GetDoseFunc(time, c(2.5, 2.5)), nextappt=time + 48))
		else if(2.3 <= inr && inr <= 3.0)
			return(list(dose=GetDoseFunc(time, c(0, 2.5)), nextappt=time + 48))
		else if(3.0 < inr)
			return(list(dose=GetDoseFunc(time, c(0, 0)), nextappt=time + 48))

	} else if(time >= 4*24 && time < 7*24) {
		
		if(dose == 10 || dose == 15) {
			
			if(inr < 2.0)
				return(list(dose=GetDoseFunc(time, c(15, 15, 15)), nextappt=time + 72))
			else if(2.0 <= inr && inr < 3.1)
				return(list(dose=GetDoseFunc(time, c(7.5, 5, 7.5)), nextappt=time + 72))
			else if(3.1 <= inr && inr <= 3.5)
				return(list(dose=GetDoseFunc(time, c(0, 5, 5)), nextappt=time + 72))
			else if(inr > 3.5)
				return(list(dose=GetDoseFunc(time, c(0, 0, 2.5)), nextappt=time + 72))
				
		} else if(dose == 5) {

			if(inr < 2.0)
				return(list(dose=GetDoseFunc(time, c(7.5, 7.5, 7.5)), nextappt=time + 72))
			else if(2.0 <= inr && inr < 3.1)
				return(list(dose=GetDoseFunc(time, c(5, 5, 5)), nextappt=time + 72))
			else if(3.1 <= inr && inr <= 3.5)
				return(list(dose=GetDoseFunc(time, c(2.5, 2.5, 2.5)), nextappt=time + 72))
			else if(inr > 3.5)
				return(list(dose=GetDoseFunc(time, c(0, 2.5, 2.5)), nextappt=time + 72))
			
		} else if(dose == 2.5) {
			
			if(inr < 2.0)
				return(list(dose=GetDoseFunc(time, c(5, 5, 5)), nextappt=time + 72))
			else if(2.0 <= inr && inr < 3.1)
				return(list(dose=GetDoseFunc(time, c(2.5, 5, 2.5)), nextappt=time + 72))
			else if(3.1 <= inr && inr <= 3.5)
				return(list(dose=GetDoseFunc(time, c(0, 2.5, 0)), nextappt=time + 72))
			else if(inr > 3.5)
				return(list(dose=GetDoseFunc(time, c(0, 0, 2.5)), nextappt=time + 72))

		} else if(dose == 0) {
			
			if(inr < 2.0)
				return(list(dose=GetDoseFunc(time, c(2.5, 2.5, 2.5)), nextappt=time + 72))
			else if(2.0 <= inr && inr < 3.1)
				return(list(dose=GetDoseFunc(time, c(2.5, 0, 2.5)), nextappt=time + 72))
			else if(3.1 <= inr && inr <= 3.5)
				return(list(dose=GetDoseFunc(time, c(0, 2.5, 0)), nextappt=time + 72))
			else if(inr > 3.5)
				return(list(dose=GetDoseFunc(time, c(0, 0, 2.5)), nextappt=time + 72))

		}
		
	} else {
	
			if(inr < 1.6)
				return(list(dose=GetDoseFunc(time, apply(as.array(c(2*lpdose, lpdose+0.1*lpdose, lpdose+0.1*lpdose, lpdose+0.1*lpdose, lpdose+0.1*lpdose)), 1, WarfDoseUp)), nextappt=time + 5*24))
			else if(1.6 <= inr && inr < 1.8)
				return(list(dose=GetDoseFunc(time, apply(as.array(numeric(7) + lpdose + 0.05*lpdose), 1, WarfDoseUp)), nextappt=time + 7*24))
			else if(1.8 <= inr && inr < 2.0) {
				
				if(dose == 0)
					dose <- WarfDoseDown(lpdose - 0.15*lpdose)
				else
					dose <- WarfDoseUp(min(10.0, dose + 0.05*dose))

				doses <- numeric(14) + dose;
				
				return(list(dose=GetDoseFunc(time, doses), nextappt=time + 14*24))
			} else if(2.0 <= inr && inr <= 3.0) {
				
				if(dose == 0)
					dose <- WarfDoseDown(lpdose - 0.15*lpdose);
				
				doses <- numeric(14) + dose;
				
				return(list(dose=GetDoseFunc(time, doses), nextappt=time + 14*24))
			} else if(3.0 < inr && inr < 3.4) {
				
				if(dose == 0)
					dose <- WarfDoseDown(lpdose - 0.15*lpdose)
				else
					dose <- WarfDoseDown(dose - 0.05*dose)
				
				doses <- numeric(14) + dose;
				
				return(list(dose=GetDoseFunc(time, doses), nextappt=time + 14*24))										
			} else if(3.4 <= inr && inr < 5) {

				doses <- numeric(7) + WarfDoseDown(lpdose - 0.1*lpdose);

				if(inr < 4)
					doses[1] <- WarfDoseDown(lpdose*0.5)
				else
					doses[1] <- 0
									
				return(list(dose=GetDoseFunc(time, doses), nextappt=time + 7*24))
				
			} else if(inr >= 5)
				return(list(dose=GetDoseFunc(time, c(0, 0)), nextappt=time + 48))
	}
};

AdjustDoseCoumagenPCx <- function(time, inr, dose, lpdose, origdose) {	
	
	if(time < 48) {
		
		return(list(dose=GetDoseFunc(time, 2*c(dose, dose)), nextappt=time + 48))
		
	} else if(time >= 48 && time < 4*24) {
				
		pcxcoeff <- origdose / 5;
				
		if(inr < 1.3)
			return(list(dose=GetDoseFunc(time, pcxcoeff*c(15, 15)), nextappt=time + 48))
		else if(1.3 <= inr && inr < 1.5)
			return(list(dose=GetDoseFunc(time, pcxcoeff*c(10, 10)), nextappt=time + 48))
		else if(1.5 <= inr && inr < 1.7)
			return(list(dose=GetDoseFunc(time, pcxcoeff*c(10, 5)), nextappt=time + 48))
		else if(1.7 <= inr && inr < 2.0)
			return(list(dose=GetDoseFunc(time, pcxcoeff*c(5, 5)), nextappt=time + 48))
		else if(2.0 <= inr && inr < 2.3)
			return(list(dose=GetDoseFunc(time, pcxcoeff*c(2.5, 2.5)), nextappt=time + 48))
		else if(2.3 <= inr && inr <= 3.0)
			return(list(dose=GetDoseFunc(time, pcxcoeff*c(0, 2.5)), nextappt=time + 48))
		else if(3.0 < inr)
			return(list(dose=GetDoseFunc(time, pcxcoeff*c(0, 0)), nextappt=time + 48))

	} else if(time >= 4*24 && time < 7*24) {

		pcxcoeff <- origdose / 5;
		dose <- c(10, 15, 5, 2.5, 0)[which.min(abs(dose * (1/pcxcoeff) - c(10, 15, 5, 2.5, 0)))];
		
		if(dose == 10 || dose == 15) {
			
			if(inr < 2.0)
				return(list(dose=GetDoseFunc(time, pcxcoeff*c(15, 15, 15)), nextappt=time + 72))
			else if(2.0 <= inr && inr < 3.1)
				return(list(dose=GetDoseFunc(time, pcxcoeff*c(7.5, 5, 7.5)), nextappt=time + 72))
			else if(3.1 <= inr && inr <= 3.5)
				return(list(dose=GetDoseFunc(time, pcxcoeff*c(0, 5, 5)), nextappt=time + 72))
			else if(inr > 3.5)
				return(list(dose=GetDoseFunc(time, pcxcoeff*c(0, 0, 2.5)), nextappt=time + 72))
				
		} else if(dose == 5) {

			if(inr < 2.0)
				return(list(dose=GetDoseFunc(time, pcxcoeff*c(7.5, 7.5, 7.5)), nextappt=time + 72))
			else if(2.0 <= inr && inr < 3.1)
				return(list(dose=GetDoseFunc(time, pcxcoeff*c(5, 5, 5)), nextappt=time + 72))
			else if(3.1 <= inr && inr <= 3.5)
				return(list(dose=GetDoseFunc(time, pcxcoeff*c(2.5, 2.5, 2.5)), nextappt=time + 72))
			else if(inr > 3.5)
				return(list(dose=GetDoseFunc(time, pcxcoeff*c(0, 2.5, 2.5)), nextappt=time + 72))
			
		} else if(dose == 2.5) {
			
			if(inr < 2.0)
				return(list(dose=GetDoseFunc(time, pcxcoeff*c(5, 5, 5)), nextappt=time + 72))
			else if(2.0 <= inr && inr < 3.1)
				return(list(dose=GetDoseFunc(time, pcxcoeff*c(2.5, 5, 2.5)), nextappt=time + 72))
			else if(3.1 <= inr && inr <= 3.5)
				return(list(dose=GetDoseFunc(time, pcxcoeff*c(0, 2.5, 0)), nextappt=time + 72))
			else if(inr > 3.5)
				return(list(dose=GetDoseFunc(time, pcxcoeff*c(0, 0, 2.5)), nextappt=time + 72))

		} else if(dose == 0) {
			
			if(inr < 2.0)
				return(list(dose=GetDoseFunc(time, pcxcoeff*c(2.5, 2.5, 2.5)), nextappt=time + 72))
			else if(2.0 <= inr && inr < 3.1)
				return(list(dose=GetDoseFunc(time, pcxcoeff*c(2.5, 0, 2.5)), nextappt=time + 72))
			else if(3.1 <= inr && inr <= 3.5)
				return(list(dose=GetDoseFunc(time, pcxcoeff*c(0, 2.5, 0)), nextappt=time + 72))
			else if(inr > 3.5)
				return(list(dose=GetDoseFunc(time, pcxcoeff*c(0, 0, 2.5)), nextappt=time + 72))

		}
		
	} else {
	
			if(inr < 1.6)
				return(list(dose=GetDoseFunc(time, apply(as.array(c(2*lpdose, lpdose+0.1*lpdose, lpdose+0.1*lpdose, lpdose+0.1*lpdose, lpdose+0.1*lpdose)), 1, WarfDoseUp)), nextappt=time + 5*24))
			else if(1.6 <= inr && inr < 1.8)
				return(list(dose=GetDoseFunc(time, numeric(7) + WarfDoseUp(lpdose + 0.05*lpdose)), nextappt=time + 7*24))
			else if(1.8 <= inr && inr < 2.0) {
				
				if(dose == 0)
					dose <- WarfDoseDown(lpdose - 0.15*lpdose)
				else
					dose <- WarfDoseUp(min(10.0, dose + 0.05*dose))

				doses <- numeric(14) + dose;
				
				return(list(dose=GetDoseFunc(time, doses), nextappt=time + 14*24))
			} else if(2.0 <= inr && inr <= 3.0) {
				
				if(dose == 0)
					dose <- WarfDoseDown(lpdose - 0.15*lpdose);
				
				doses <- numeric(14) + dose;
				
				return(list(dose=GetDoseFunc(time, doses), nextappt=time + 14*24))
			} else if(3.0 < inr && inr < 3.4) {
				
				if(dose == 0)
					dose <- WarfDoseDown(lpdose - 0.15*lpdose)
				else
					dose <- WarfDoseDown(dose - 0.05*dose)
				
				doses <- numeric(14) + dose;
				
				return(list(dose=GetDoseFunc(time, doses), nextappt=time + 14*24))										
			} else if(3.4 <= inr && inr < 5) {

				doses <- numeric(7) + WarfDoseDown(lpdose - 0.1*lpdose);

				if(inr < 4)
					doses[1] <- WarfDoseDown(lpdose*0.5)
				else
					doses[1] <- 0
									
				return(list(dose=GetDoseFunc(time, doses), nextappt=time + 7*24))
				
			} else if(inr >= 5)
				return(list(dose=GetDoseFunc(time, c(0, 0)), nextappt=time + 48))
	}
};

RunSample <- function(age, cyp2c9, vkorc1, startdose, adjuster) {
		
	# constants from Hamberg et al.
	INRmax <- 20;
	
	thetaV1 <- 13.8;
	thetaKaS <- 2;
	thetaQ <- 0.131;
	thetaV2 <- 6.59;
	thetaAGECLs <- -0.0091;       # the minus sign is the authors' post-pub correction to the published constant
	thetaCLs <- 0.314;
	theta11 <- 0;
	theta12 <- 0.315;
	theta13 <- 0.453;
	theta22 <- 0.722;
	theta23 <- 0.690;
	theta33 <- 0.852;
	
	omegaCLS <- 0.310;
	omegaV1 <- 0.262;
	omegaV2 <- 0.991;
	
	sigmaSs <- 0.0908;
	sigmaSss <- 0.301;

	# default to wild type	
	thetaCyp2c9 <- theta11;
	
	if(cyp2c9 == "*1/*2") {
		thetaCyp2c9 <- theta12;
	}
	if(cyp2c9 == "*1/*3") {
		thetaCyp2c9 <- theta13;
	}
	if(cyp2c9 == "*2/*2") {
		thetaCyp2c9 <- theta22;
	}
	if(cyp2c9 == "*2/*3") {
		thetaCyp2c9 <- theta23;
	}
	if(cyp2c9 == "*3/*3") {
		thetaCyp2c9 <- theta33;
	}
	
	# compartment volumes sampled as described in Hamberg et al.
	nuV1 <- rtnorm(1, mean = 0, sd = omegaV1, lower = qnorm(0.25, mean = 0, sd = omegaV1), upper = qnorm(0.75, mean = 0, sd = omegaV1));
	nuV2 <- rtnorm(1, mean = 0, sd = omegaV2, lower = qnorm(0.25, mean = 0, sd = omegaV2), upper = qnorm(0.75, mean = 0, sd = omegaV2));
	nuCLsi <- rtnorm(1, mean = 0, sd = omegaCLS, lower = qnorm(0.25, mean = 0, sd = omegaCLS), upper = qnorm(0.75, mean = 0, sd = omegaCLS));
	
	V1 <- thetaV1*exp(nuV1);
	V2 <- thetaV2*exp(nuV2);
	
	# clearance depends on age and Cyp2c9
	CLsi <- thetaCLs*(1 + thetaAGECLs*(age-71))*(1-thetaCyp2c9)*exp(nuCLsi);
	
	# inter-compartment micro-constants computed from Q (inter-compartment clearance)
	# by assuming Q = k12 * V1 = k21 * V2
	k12 <- thetaQ / V1;
	k21 <- V1/V2*k12;
	
	kel <- CLsi / V1;
	halflife <- 0.693/kel;
	
	# concentration calculation as described on boomer.org and Bauer's book
	beta <- 0.5*(k12 + k21 + kel - sqrt((k12 + k21 + kel)^2 - 4*k21*kel));
	alpha <- (k21*kel)/beta;
	
	A <- (thetaKaS/V1) * ((k21 - alpha)/((thetaKaS - alpha)*(beta - alpha)));
	B <- (thetaKaS/V1) * ((k21 - beta)/((thetaKaS - beta)*(alpha - beta)));
	
	Csbase <- function(t, D, td) {		
		if(t - td < 0) {
			0
		} else {
			return(D*(A*exp(-1*alpha*(t-td)) + B*exp(-1*beta*(t-td)) - (A+B)*exp(-1*thetaKaS*(t-td))));
		}
	};

	# more constants for the PD model from Hamberg et al.
	thetaEmax <- 1.0;
	thetagamma <- 0.424;
	thetaGG <- 4.61;
	thetaAG <- 3.02;
	thetaAA <- 2.20;
	thetaMTT1 <- 11.6;
	thetaMTT2 <- 120;
	thetalambda <- 3.61;
	
	# default to wild type
	thetaVkORC1 <- thetaGG;
	
	if(vkorc1 == "A/G") {
		thetaVkORC1 <- thetaAG;
	}
	if(vkorc1 == "A/A") {
		thetaVkORC1 <- thetaAA;
	}
	
	# mean transit times and EC50 sampled from constants in Hamberg's paper
	omegaMTT1 <- 0.141;
	omegaMTT2 <- 1.02;
	omegaEC50 <- 0.409;
	
	sigmaINR <- 0.0325;
	
	nuMTT1 <- rtnorm(1, mean = 0, sd = omegaMTT1, lower = qnorm(0.25, mean = 0, sd = omegaMTT1), upper = qnorm(0.75, mean = 0, sd = omegaMTT1));
	nuMTT2 <- rtnorm(1, mean = 0, sd = omegaMTT2, lower = qnorm(0.25, mean = 0, sd = omegaMTT2), upper = qnorm(0.75, mean = 0, sd = omegaMTT2));
	nuEC50 <- rtnorm(1, mean = 0, sd = omegaEC50, lower = qnorm(0.25, mean = 0, sd = omegaEC50), upper = qnorm(0.75, mean = 0, sd = omegaEC50));
	
	gamma <- thetagamma;
	lambda <- thetalambda;
	
	MTT1 <- thetaMTT1*exp(nuMTT1);
	MTT2 <- thetaMTT2;
	EC50 <- thetaVkORC1*exp(nuEC50);
	
	interval <- 1;
	tlength <- 90*24;	
	
	# ktr2 = 6 / MTT2 is from a post-publication correction to Hamberg's model
	ktr1 <- 1 / MTT1;
	ktr2 <- 6 / MTT2;
	
	# this function actually computes the concentration from a dose history, baseline INR, steady-state indicator,
	# and base time (this is the time at which the first dose in the doses array was administered)
	GetINRFunction <- function(doses, base, basetime, nextappt, Ainit) {
		
		# use linear superposition to account for accumulated concentration from previous doses (from Fusaro et al.)
		# this assumes that each dose acts independently, and the extent of absoprtion and clearance are the same
		# for each dose.
		Cs <- function(t) {
			# bioavailability = 90%, reduce dose by 1/2 because we only consider the effect of S-warfarin
			bases <- apply(as.array(1:length(doses)), 1, function(i) Csbase(t, 0.9*0.5*doses[i], (i-1)*24));			
			perturbs <- NA;
			if(t >= halflife*6) {
				perturbs <- exp(-1*rtnorm(length(doses), mean = 0, sd = sigmaSs));
			} else {
				perturbs <- exp(-1*rtnorm(length(doses), mean = 0, sd = sigmaSss));
			}
			
			return(sum(bases*perturbs));
		};
				
		Csvals <- apply(as.array(seq(from = basetime, to = nextappt, by = interval)), 1, function(t) { csv <- Cs(t); 1 - (thetaEmax*csv^gamma) / (EC50^gamma + csv^gamma) });
		trvals <- c(ktr1, ktr1, ktr1, ktr1, ktr1, ktr1, ktr2);
				
		compartments <- function(t, A, params) {
			
			v <- numeric(7) + Csvals[((t-basetime)/interval)+1];

			v[2:6] <- A[1:5];
			v[1:7] <- trvals*(v[1:7] - A[1:7]);
			
			return(list(v));			
		};
		
		# solve the ode's
		# we solve for each hour in the trial (90 days)
		sol <- rk(times = seq(from = basetime, to = nextappt, by = interval), y = Ainit, func = compartments, parms = NULL);
		
		# return a function value that has the ode solution in scope
		inrfunc <- function(t) {0
			epsilonINR <- rtnorm(1, mean = 0, sd = sigmaINR);
			(base + INRmax*(1 - sol[(t-basetime)/interval+1,7][1]*sol[(t-basetime)/interval+1,8][1])^lambda)*exp(-1*epsilonINR)
		};		
		
		return(list(finr = inrfunc, state = as.numeric(sol[(nextappt-basetime)/interval+1,2:8])));
	}
	
	# now we begin to run the simulated clinical trial	
	
	dosecur <- startdose;
	doses <- c();
	alldoses <- c();
	# the initial INR is a normal variate around 1
	# embarrassingly, i got the sd from Wikipedia's description of INR: "the INR in absence of anticoagulation therapy is 0.8-1.2"
	# if we can find a better source for this, we should. i tried to find some published statistics, but came up dry.
	inrappt <- rtnorm(1, mean = 1.0, sd = 0.0879217, lower = qnorm(0.25, mean = 1.0, sd = 0.0879217), upper = qnorm(0.75, mean = 1.0, sd = 0.0879217));
	inrdata <- c();
	lpdose <- startdose;
	
	basestroke <- ifelse(runif(1, 0, 1) >= 0.18, 0.00012614, 0.000316144);
	
	apptdata <- adjuster(0, inrappt, dosecur, lpdose, startdose);
	dosefun <- apptdata$dose;
	nextappt <- apptdata$nextappt;
	
	inrcomps <- GetINRFunction(apply(as.array(seq(from=0, to=nextappt-24, by = 24)), 1, dosefun), inrappt, 0, nextappt, numeric(7)+1);
	Finr <- inrcomps$finr;
	Ainit <- inrcomps$state;
	
	logitStroke <- function(inr) {
		return(-0.38 - (3.52 * min(inr, 2.0)) + (0.68  *max(inr, 3.0)));
	};
	logitICH <- function(inr) {
		return(-8.93 + (1.67 * max(inr, 3.0)));
	}
	strokerisk <- function(base, inr) {
		rr <- ifelse(inr < 2.0, 0.46*5.14, ifelse(inr > 3.0, 1.6*0.46, 0.46));
		return(base*rr);
	}
	ichrisk <- function(inr) {		
		base <- 6.85787e-6;
		return(ifelse(inr < 2.0, 0.92*base, ifelse(inr > 3.0, 4.31*base, base)));
	}
	echrisk <- function(inr) {		
		base <- 5.48492e-6;
		return(ifelse(inr < 2.0, 2.14*base, ifelse(inr > 3.0, 5.88*base, 2.22*base)));
	}
	deathrisk <- function(stbase, inr) {
		strisk <- strokerisk(stbase, inr);
		irisk <- ichrisk(inr);
		erisk <- echrisk(inr);
		
		sdeath <- ifelse(inr < 2.0, 0.175*strisk, 0.081*strisk);
		
		return(sdeath + irisk*0.516 + erisk*0.0147);
	}
	
	#print(sprintf("baseline inr = %f", inrappt));
	
	strisk <- 0.0;
	icrisk <- 0.0;
	ecrisk <- 0.0;
	drisk <- 0.0;
	inrcur <- inrappt;
	
	# proceed in 24-hour increments
	for(time in seq(from = 0, to = tlength, by = 24)) {				
				
		#print(sprintf("inr: %f, st: %f, ich: %f, ech: %f, death: %f", inrcur, strisk, icrisk, ecrisk, drisk));
		
		if(nextappt <= time) {
			
			inrcur <- Finr(time);
			strisk <- strisk + 365*strokerisk(basestroke, inrcur);
			icrisk <- icrisk + 365*ichrisk(inrcur);
			ecrisk <- ecrisk + 365*echrisk(inrcur);
			drisk <- drisk + 365*deathrisk(basestroke, inrcur);			
			
			# at each appointment, get the current INR
			inrdata <- append(inrdata, inrcur);
			# assume we're at stable concentration after the first week of therapy
			
			#print(sprintf("    inr = %f, dose = %f, time = %d", inrcur, dosecur, time))			
			
			# get the next dose and appointment time
			apptdata <- adjuster(time, inrcur, dosecur, lpdose, startdose);

			dosefun <- apptdata$dose;
			nextappt <- apptdata$nextappt;			
			
			inrcomps <- GetINRFunction(append(alldoses, apply(as.array(seq(from=time, to=nextappt-24, by=24)), 1, dosefun)), inrappt, time, nextappt, Ainit);
			Finr <- inrcomps$finr;
			Ainit <- inrcomps$state;
		} 
		
		#print(sprintf("time = %d, dose = %f, inr = %f", time, dosefun(time), Finr(time)));
		
		dosecur <- dosefun(time);
		if(dosecur > 0) lpdose <- dosecur;

		# keep track of all doses in case we want to track some stats on these later
		alldoses <- append(alldoses, dosecur);				
		
	}
	
	# we just take the stable dose to be whatever the most recent dose was
	stabledose <- alldoses[length(alldoses)];
	ttr <- sum(inrdata >= 1.8 & inrdata <= 3.2) / length(inrdata);
	
	#print(sprintf("stroke risk: %f, ich risk: %f, ech risk: %f, death risk: %f", strisk, icrisk, ecrisk, drisk));
	#print(sprintf("stroke risk: %f (dice: %f), ich risk: %f (dice: %f)", strisk, runif(1,0,100) <= 100*strisk, icrisk, runif(1,0,100) <= 100*icrisk));
	
	#print(as.numeric(inrdata));
	
	# return the stable dose and the time in therapeutic range
	return(list(dose=stabledose, meanttr=ttr, stroke=strisk, ich=icrisk, ech=ecrisk, death=drisk));
}

RunDataSample <- function(sample, dose, adjuster) {
	
	age <- as.numeric(25 + 10*(sample["age"]-1));
	
	cyp2c9 <- "*1/*1";
	if(sample["cyp2c9=12"] > 0) cyp2c9 <- "*1/*2";
	if(sample["cyp2c9=13"] > 0) cyp2c9 <- "*1/*3";
	if(sample["cyp2c9=22"] > 0) cyp2c9 <- "*2/*2";
	if(sample["cyp2c9=23"] > 0) cyp2c9 <- "*2/*3";
	if(sample["cyp2c9=33"] > 0) cyp2c9 <- "*3/*3";
	
	vkorc1 <- "G/G";
	if(sample["vkorc1=CT"] > 0) vkorc1 <- "A/G";
	if(sample["vkorc1=TT"] > 0) vkorc1 <- "A/A";
	
	#print(sprintf("age = %f, cyp2c9 = %s, vkorc1 = %s, dose = %f", age, cyp2c9, vkorc1, dose));
	
	RunSample(age, cyp2c9, vkorc1, dose, adjuster)
}

RunGivenSamplesLabeled <- function(datafile, labelfile, n, start) {
	
	data <- read.csv(datafile);
	labels <- read.csv(labelfile);
	ttrs <- c();
	doses <- c();
	
	for(i in (start):(start+n-1)) {
		
		set.seed(i);
		
		label <- labels[i,1];
		dose <- 5;
		if(label == 1)
			dose <- 3;
		if(label == 2)
			dose <- 5;
		if(label == 3)
			dose <- 7;
		
		cur <- RunDataSample(data[i,], dose)
		ttrs <- append(ttrs, cur[2]);
		doses <- append(doses, cur[1]);
		print(sprintf("    ----(i = %d) cur ttr = %f, mean ttr = %f, doses = %f", i, cur[2], mean(ttrs), mean(doses)));
		print("");
	}
	
	data.frame(ttrs,doses)
}

RunGivenSamplesFixed <- function(data, dose, n, start) {
	
	ttrs <- c();
	doses <- c();

	for(i in (start):(start+n-1)) {
		
		set.seed(i);
				
		cur <- RunDataSample(data[i,], dose)
		ttrs <- append(ttrs, cur[2]);
		doses <- append(doses, cur[1]);
		print(sprintf("    ----(i = %d) cur ttr = %f, mean ttr = %f, mean doses = %f", i, cur[2], mean(ttrs), mean(doses)));
		print("");
	}
	
	data.frame(ttrs,doses)	
}

ci <- function(data) {
	data <- as.numeric(data);
	if(length(data)<=1) {
		return(0);
	} else {
		return(qt(0.975, df=length(data)-1)*sd(data)/sqrt(length(data)));
	}
}

RunGivenSamplesModel <- function(data, model, n, start, adjuster) {
	
	#scale <- model$scale;
	#model <- model$model;
	
	ttrs <- c();	
	strokes <- c();
	ichs <- c();
	echs <- c();
	deaths <- c();
	
	ni <- 0;
	
	for(i in (start):(start+n-1)) {
		
		set.seed(i);
		
		ni <- ni + 1;
		
		scaled <- data[i,];
		modelval <- model(scaled);
				
		#scaled <- data[i,] * scale;
		#modelval <- as.numeric((1/scale["dose"]) * model(scaled[!(names(scaled) %in% "dose")]));
		modelval <- ifelse(modelval <= 0.5, 0.5, modelval);
		dose <- WarfDose(modelval);
		#dose <- WarfDose(as.numeric(data[i,"dose"]));
		
		#print(dose);
		
		cur <- RunDataSample(data[i,], dose, adjuster)
		ttrs <- append(ttrs, cur$meanttr);
		deaths <- append(deaths, cur$death);
		strokes <- append(strokes, cur$stroke);
		ichs <- append(ichs, cur$ich);
		echs <- append(echs, cur$ech);
		#if(i %% 50 == 1) print(sprintf("    ----(i = %d) cur ttr = %f, mean ttr = %f, doses = %f", i, cur[2], mean(ttrs), mean(doses)));
		print(sprintf("    ----(i = %d) cur ttr = %f, mean ttr = %f (ci = %f), stroke: %f (ci = %f), bleed: %f (ci = %f), death: %f (ci = %f)", i, cur[2], mean(ttrs), ci(ttrs), sum(strokes), ni*ci(strokes), sum(ichs+echs), ni*ci(ichs+echs), sum(deaths), ni*ci(deaths)));
		#print("");
	}
	
	return(list(strokes=as.numeric(strokes), bleeds=as.numeric(sum(ichs+echs)), deaths=as.numeric(deaths), ttrs=as.numeric(ttrs)));
	
	#return(list(strokes=as.numeric(sum(strokes)), bleeds=as.numeric(sum(ichs+echs)), deaths=as.numeric(sum(deaths)), meanttr=mean(ttrs)), strokeci=n*ci(strokes), bleedci=n*ci(ichs+echs), deathci=n*ci(deaths), ttrci=ci(ttrs));
}

RunManyModelsTrial <- function(training, data, eps, nmodels, n, start, adjuster) {

	ttrs <- c();	
	strokes <- c();
	ichs <- c();
	echs <- c();
	deaths <- c();
	
	ni <- 0;
	
	model <- diffpLR(training, "dose", eps);
	
	for(i in (start):(start+n-1)) {
		
		if(i %% round(n/nmodels) == 0 && eps > 0) {
		
			print(sprintf("training new epsilon=%f", eps));	
			model <- diffpLR(training, "dose", eps);
		}
		
		set.seed(i);
		
		ni <- ni + 1;
		
		scaled <- data[i,];
		modelval <- model(scaled);				
		modelval <- ifelse(modelval <= 0.5, 0.5, modelval);
		dose <- WarfDose(modelval);
				
		cur <- RunDataSample(data[i,], dose, adjuster)
		ttrs <- append(ttrs, cur$meanttr);
		deaths <- append(deaths, cur$death);
		strokes <- append(strokes, cur$stroke);
		ichs <- append(ichs, cur$ich);
		echs <- append(echs, cur$ech);

		print(sprintf("    ----(i = %d) cur ttr = %f, mean ttr = %f (ci = %f), stroke: %f (ci = %f), bleed: %f (ci = %f), death: %f (ci = %f)", i, cur[2], mean(ttrs), ci(ttrs), sum(strokes), ni*ci(strokes), sum(ichs+echs), ni*ci(ichs+echs), sum(deaths), ni*ci(deaths)));

	}
	
	return(list(strokes=as.numeric(strokes), bleeds=as.numeric(ichs+echs), deaths=as.numeric(deaths), ttrs=as.numeric(ttrs)));
}

RunManyDatasetsTrial <- function(fileprefix, data, nmodels, n, start, adjuster) {

	ttrs <- c();	
	strokes <- c();
	ichs <- c();
	echs <- c();
	deaths <- c();
	
	ni <- 0;
	dataset <- 1;

	filename <- paste0(fileprefix, "-", as.character(dataset), ".csv");
	
	print(sprintf("loading %s", filename));
	
	training <- read.csv(filename);
	names(training) <- names(data);
	training <- as.matrix(training);
	model <- diffpLR(training, "dose", 0);
				
	for(i in (start):(start+n-1)) {
		
		if(i %% round(n/nmodels) == 0) {
			
			dataset <- dataset + 1;
			
			filename <- paste0(fileprefix, "-", as.character(dataset), ".csv");
			
			print(sprintf("loading %s", filename));
			
			training <- read.csv(filename);
			names(training) <- names(data);
			training <- as.matrix(training);
			model <- diffpLR(training, "dose", 0);
		}
		
		set.seed(i);
		
		ni <- ni + 1;
		
		scaled <- data[i,];
		modelval <- model(scaled);				
		modelval <- ifelse(modelval <= 0.5, 0.5, modelval);
		dose <- WarfDose(modelval);
				
		cur <- RunDataSample(data[i,], dose, adjuster)
		ttrs <- append(ttrs, cur$meanttr);
		deaths <- append(deaths, cur$death);
		strokes <- append(strokes, cur$stroke);
		ichs <- append(ichs, cur$ich);
		echs <- append(echs, cur$ech);

		print(sprintf("    ----(i = %d) cur ttr = %f, mean ttr = %f (ci = %f), stroke: %f (ci = %f), bleed: %f (ci = %f), death: %f (ci = %f)", i, cur[2], mean(ttrs), ci(ttrs), sum(strokes), ni*ci(strokes), sum(ichs+echs), ni*ci(ichs+echs), sum(deaths), ni*ci(deaths)));

	}
	
	return(list(strokes=as.numeric(strokes), bleeds=as.numeric(ichs+echs), deaths=as.numeric(deaths), ttrs=as.numeric(ttrs)));
}

t.test.silent <- function(...) { 
	obj<-try(t.test(...), silent=TRUE) 
	if (is(obj, "try-error")) return(NA) else return(obj$p.value); 
}

GetPValues <- function(ttrs, strokes, bleeds, deaths, vttrs, vstrokes, vbleeds, vdeaths) {

	pvals <- matrix(data=0, nrow=4, ncol=6);
	rownames(pvals) <- c("ttrs", "strokes", "bleeds", "deaths");
	colnames(pvals) <- c("gt", "lt", "neq", "mean", "var", "rr");

	# ttrs > vttrs
	pvals["ttrs","gt"] <- t.test.silent(x=ttrs, y=vttrs, alternative="g", conf.level=0.95, paired=TRUE);
	# strokes < vstrokes
	pvals["strokes","lt"] <- t.test.silent(x=strokes, y=vstrokes, alternative="l", conf.level=0.95, paired=TRUE);
	# bleeds < vbleeds
	pvals["bleeds","lt"] <- t.test.silent(x=bleeds, y=vbleeds, alternative="l", conf.level=0.95, paired=TRUE);
	# deaths < vdeaths
	pvals["deaths","lt"] <- t.test.silent(x=deaths, y=vdeaths, alternative="l", conf.level=0.95, paired=TRUE);
	# ttrs < vttrs
	pvals["ttrs","lt"] <- t.test.silent(x=ttrs, y=vttrs, alternative="l", conf.level=0.95, paired=TRUE);
	# strokes > vstrokes
	pvals["strokes","gt"] <- t.test.silent(x=strokes, y=vstrokes, alternative="g", conf.level=0.95, paired=TRUE);
	# bleeds > vbleeds
	pvals["bleeds","gt"] <- t.test.silent(x=bleeds, y=vbleeds, alternative="g", conf.level=0.95, paired=TRUE);
	# deaths > vdeaths
	pvals["deaths","gt"] <- t.test.silent(x=deaths, y=vdeaths, alternative="g", conf.level=0.95, paired=TRUE);
	# ttrs != vttrs
	pvals["ttrs","neq"] <- t.test.silent(x=ttrs, y=vttrs, alternative="t", conf.level=0.95, paired=TRUE);
	# strokes != vstrokes
	pvals["strokes","neq"] <- t.test.silent(x=strokes, y=vstrokes, alternative="t", conf.level=0.95, paired=TRUE);
	# bleeds != vbleeds
	pvals["bleeds","neq"] <- t.test.silent(x=bleeds, y=vbleeds, alternative="t", conf.level=0.95, paired=TRUE);
	# deaths != vdeaths
	pvals["deaths","neq"] <- t.test.silent(x=deaths, y=vdeaths, alternative="t", conf.level=0.95, paired=TRUE);	

	pvals["ttrs","mean"] <- mean(ttrs);
	pvals["strokes","mean"] <- mean(strokes);
	pvals["bleeds","mean"] <- mean(bleeds);
	pvals["deaths","mean"] <- mean(deaths);
	pvals["ttrs","var"] <- var(ttrs);
	pvals["strokes","var"] <- var(strokes);
	pvals["bleeds","var"] <- var(bleeds);
	pvals["deaths","var"] <- var(deaths);
	pvals["ttrs","rr"] <- mean(ttrs-vttrs);
	pvals["strokes","rr"] <- mean(strokes/vstrokes)
	pvals["bleeds","rr"] <- mean(bleeds/vbleeds);
	pvals["deaths","rr"] <- mean(deaths/vdeaths);	
	
	
	#summ <- sprintf("ttrs.gt=%f (mean.ttrs=%f) strokes.lt=%f (mean.strokes=%f) bleeds.lt=%f (mean.bleeds=%f) deaths.lt=%f (mean.deaths=%f)", ttrs.gt, mean(ttrs), strokes.lt, mean(strokes), bleeds.lt, mean(bleeds), deaths.lt, mean(deaths));
	
	return(pvals);
}

RunComparisonTrials <- function(training, validation, nmodels, eps) {
	
	ttrs.fixed <- c();	
	strokes.fixed <- c();
	ichs.fixed <- c();
	echs.fixed <- c();
	deaths.fixed <- c();
	ttrs.lr <- c();	
	strokes.lr <- c();
	ichs.lr <- c();
	echs.lr <- c();
	deaths.lr <- c();
	ttrs.dplr <- c();	
	strokes.dplr <- c();
	ichs.dplr <- c();
	echs.dplr <- c();
	deaths.dplr <- c();
	
	clfun.lr <- diffpLR.reverse.functional(training, "dose", eps, vkoattr)$forward;
	clfun.dplr <- diffpLR.reverse.functional(training, "dose", 0, vkoattr)$forward;

	for(i in 1:nrow(validation)) {				
		
		cat(sprintf("\ni=%d, epsilon=%f\n\n", i, eps));
		
		if(i %% max(1,round(nrow(validation)/nmodels)) == 0) {									
			clfun.dplr <- diffpLR.reverse.functional(training, "dose", eps, vkoattr)$forward;
		}
						
		scaled <- validation[i,];
		
		# fixed-dose
		set.seed(i);
		dose.fixed <- 5.0;
		cur.fixed <- RunDataSample(scaled, dose.fixed, AdjustDoseCoumagen);
		ttrs.fixed <- append(ttrs.fixed, cur.fixed$meanttr);
		deaths.fixed <- append(deaths.fixed, cur.fixed$death);
		strokes.fixed <- append(strokes.fixed, cur.fixed$stroke);
		ichs.fixed <- append(ichs.fixed, cur.fixed$ich);
		echs.fixed <- append(echs.fixed, cur.fixed$ech);
		
		cat(sprintf("----FIXED: ttrs=%.4f strokes=%.4f bleeds=%.4f deaths=%.4f\n", mean(ttrs.fixed), mean(strokes.fixed), mean(ichs.fixed+echs.fixed), mean(deaths.fixed)));	

		# non-private lr
		set.seed(i);
		dose.lr <- WarfDose(clfun.lr(scaled));				
		cur.lr <- RunDataSample(scaled, dose.lr, AdjustDoseCoumagenPCx);
		ttrs.lr <- append(ttrs.lr, cur.lr$meanttr);
		deaths.lr <- append(deaths.lr, cur.lr$death);
		strokes.lr <- append(strokes.lr, cur.lr$stroke);
		ichs.lr <- append(ichs.lr, cur.lr$ich);
		echs.lr <- append(echs.lr, cur.lr$ech);
		
		cat(sprintf("----LR:\n"));
		print(GetPValues(ttrs.lr, strokes.lr, ichs.lr+echs.lr, deaths.lr, ttrs.fixed, strokes.fixed, ichs.fixed+echs.fixed, deaths.fixed), digits=6);

		# private lr
		set.seed(i);
		dose.dplr <- WarfDose(clfun.dplr(scaled));
		cur.dplr <- RunDataSample(scaled, dose.dplr, AdjustDoseCoumagenPCx);
		ttrs.dplr <- append(ttrs.dplr, cur.dplr$meanttr);
		deaths.dplr <- append(deaths.dplr, cur.dplr$death);
		strokes.dplr <- append(strokes.dplr, cur.dplr$stroke);
		ichs.dplr <- append(ichs.dplr, cur.dplr$ich);
		echs.dplr <- append(echs.dplr, cur.dplr$ech);
		
		cat(sprintf("----DPLR:\n"));
		print(GetPValues(ttrs.dplr, strokes.dplr, ichs.dplr+echs.dplr, deaths.dplr, ttrs.fixed, strokes.fixed, ichs.fixed+echs.fixed, deaths.fixed), digits=6);

	}
	
	rawdata <- data.frame(	ttrs.fixed,strokes.fixed,ichs.fixed,echs.fixed,deaths.fixed,
							ttrs.lr,strokes.lr,ichs.lr,echs.lr,deaths.lr,
							ttrs.dplr,strokes.dplr,ichs.dplr,echs.dplr,deaths.dplr);
	pvals <- list(	lr=GetPValues(ttrs.lr, strokes.lr, ichs.lr+echs.lr, deaths.lr, ttrs.fixed, strokes.fixed, ichs.fixed+echs.fixed, deaths.fixed),
					dplr=GetPValues(ttrs.dplr, strokes.dplr, ichs.dplr+echs.dplr, deaths.dplr, ttrs.fixed, strokes.fixed, ichs.fixed+echs.fixed, deaths.fixed));

	curdir <- getwd();
	dir.create(file.path(curdir, "utility-results"), showWarnings = FALSE);
	setwd(file.path(curdir, "utility-results"));
	filename <- sprintf("runComparisonTrials-eps.%.2f-nmodels.%d-%s.Robject", eps, nmodels, format(Sys.time(), "%b%d.%H.%M.%S"));
	dput(list(rawdata=rawdata, pvals=pvals), file=filename);
	setwd(curdir);
	
	return(list(rawdata=rawdata, pvals=pvals));
}

RunComparisonTrials.xi <- function(training, validation, nmodels, eps) {
	
	ttrs.fixed <- c();	
	strokes.fixed <- c();
	ichs.fixed <- c();
	echs.fixed <- c();
	deaths.fixed <- c();
	ttrs.lr <- c();	
	strokes.lr <- c();
	ichs.lr <- c();
	echs.lr <- c();
	deaths.lr <- c();
	ttrs.dplr <- c();	
	strokes.dplr <- c();
	ichs.dplr <- c();
	echs.dplr <- c();
	deaths.dplr <- c();
		
	clfun.dplr <- diffpLR.xi.tuned(training, "dose", epsilon=eps)$clfun$forward;
	clfun.lr <- diffpLR.reverse.functional(training, "dose", 0, vkoattr)$forward;

	for(i in 1:nrow(validation)) {				
		
		cat(sprintf("\ni=%d, epsilon=%f\n\n", i, eps));
		
		if(i %% max(1,round(nrow(validation)/nmodels)) == 0) {												
			clfun.dplr <- diffpLR.xi.tuned(training, "dose", epsilon=eps)$clfun$forward;
		}
						
		scaled <- validation[i,];
		
		# fixed-dose
		set.seed(i);
		dose.fixed <- 5.0;
		cur.fixed <- RunDataSample(scaled, dose.fixed, AdjustDoseCoumagen);
		ttrs.fixed <- append(ttrs.fixed, cur.fixed$meanttr);
		deaths.fixed <- append(deaths.fixed, cur.fixed$death);
		strokes.fixed <- append(strokes.fixed, cur.fixed$stroke);
		ichs.fixed <- append(ichs.fixed, cur.fixed$ich);
		echs.fixed <- append(echs.fixed, cur.fixed$ech);
		
		cat(sprintf("----FIXED: ttrs=%.4f strokes=%.4f bleeds=%.4f deaths=%.4f\n", mean(ttrs.fixed), mean(strokes.fixed), mean(ichs.fixed+echs.fixed), mean(deaths.fixed)));	

		# non-private lr
		set.seed(i);
		dose.lr <- WarfDose(clfun.lr(scaled));				
		cur.lr <- RunDataSample(scaled, dose.lr, AdjustDoseCoumagenPCx);
		ttrs.lr <- append(ttrs.lr, cur.lr$meanttr);
		deaths.lr <- append(deaths.lr, cur.lr$death);
		strokes.lr <- append(strokes.lr, cur.lr$stroke);
		ichs.lr <- append(ichs.lr, cur.lr$ich);
		echs.lr <- append(echs.lr, cur.lr$ech);
		
		cat(sprintf("----LR:\n"));
		print(GetPValues(ttrs.lr, strokes.lr, ichs.lr+echs.lr, deaths.lr, ttrs.fixed, strokes.fixed, ichs.fixed+echs.fixed, deaths.fixed), digits=6);

		# private lr
		set.seed(i);
		dose.dplr <- WarfDose(clfun.dplr(scaled));
		cur.dplr <- RunDataSample(scaled, dose.dplr, AdjustDoseCoumagenPCx);
		ttrs.dplr <- append(ttrs.dplr, cur.dplr$meanttr);
		deaths.dplr <- append(deaths.dplr, cur.dplr$death);
		strokes.dplr <- append(strokes.dplr, cur.dplr$stroke);
		ichs.dplr <- append(ichs.dplr, cur.dplr$ich);
		echs.dplr <- append(echs.dplr, cur.dplr$ech);
		
		cat(sprintf("----DPLR:\n"));
		print(GetPValues(ttrs.dplr, strokes.dplr, ichs.dplr+echs.dplr, deaths.dplr, ttrs.fixed, strokes.fixed, ichs.fixed+echs.fixed, deaths.fixed), digits=6);

	}
	
	rawdata <- data.frame(	ttrs.fixed,strokes.fixed,ichs.fixed,echs.fixed,deaths.fixed,
							ttrs.lr,strokes.lr,ichs.lr,echs.lr,deaths.lr,
							ttrs.dplr,strokes.dplr,ichs.dplr,echs.dplr,deaths.dplr);
	pvals <- list(	lr=GetPValues(ttrs.lr, strokes.lr, ichs.lr+echs.lr, deaths.lr, ttrs.fixed, strokes.fixed, ichs.fixed+echs.fixed, deaths.fixed),
					dplr=GetPValues(ttrs.dplr, strokes.dplr, ichs.dplr+echs.dplr, deaths.dplr, ttrs.fixed, strokes.fixed, ichs.fixed+echs.fixed, deaths.fixed));

	curdir <- getwd();
	dir.create(file.path(curdir, "utility-results"), showWarnings = FALSE);
	setwd(file.path(curdir, "utility-results"));
	filename <- sprintf("runComparisonTrials-eps.%.2f-nmodels.%d-%s.Robject", eps, nmodels, format(Sys.time(), "%b%d.%H.%M.%S"));
	dput(list(rawdata=rawdata, pvals=pvals), file=filename);
	setwd(curdir);
	
	return(list(rawdata=rawdata, pvals=pvals));
}

RunComparisonTrials.xi.theoretical <- function(training, validation, nmodels, eps) {
	
	ttrs.fixed <- c();	
	strokes.fixed <- c();
	ichs.fixed <- c();
	echs.fixed <- c();
	deaths.fixed <- c();
	ttrs.lr <- c();	
	strokes.lr <- c();
	ichs.lr <- c();
	echs.lr <- c();
	deaths.lr <- c();
	ttrs.dplr <- c();	
	strokes.dplr <- c();
	ichs.dplr <- c();
	echs.dplr <- c();
	deaths.dplr <- c();

    d <- ncol(training)-1;
    n <- nrow(training);	

    param.lambda = sqrt(d/(n*eps));

    clfun.dplr <- diffpLR.reverse.xi(training, "dose", vkoattr, epsilon=eps, R=1, lambda=param.lambda)$forward;	
	clfun.lr <- diffpLR.reverse.functional(training, "dose", 0, vkoattr)$forward;

	for(i in 1:nrow(validation)) {				
		
		cat(sprintf("\ni=%d, epsilon=%f\n\n", i, eps));
		
		if(i %% max(1,round(nrow(validation)/nmodels)) == 0) {												
			clfun.dplr <- diffpLR.reverse.xi(training, "dose", vkoattr, epsilon=eps, R=1, lambda=param.lambda)$forward;
		}
						
		scaled <- validation[i,];
		
		# fixed-dose
		set.seed(i);
		dose.fixed <- 5.0;
		cur.fixed <- RunDataSample(scaled, dose.fixed, AdjustDoseCoumagen);
		ttrs.fixed <- append(ttrs.fixed, cur.fixed$meanttr);
		deaths.fixed <- append(deaths.fixed, cur.fixed$death);
		strokes.fixed <- append(strokes.fixed, cur.fixed$stroke);
		ichs.fixed <- append(ichs.fixed, cur.fixed$ich);
		echs.fixed <- append(echs.fixed, cur.fixed$ech);
		
		cat(sprintf("----FIXED: ttrs=%.4f strokes=%.4f bleeds=%.4f deaths=%.4f\n", mean(ttrs.fixed), mean(strokes.fixed), mean(ichs.fixed+echs.fixed), mean(deaths.fixed)));	

		# non-private lr
		set.seed(i);
		dose.lr <- WarfDose(clfun.lr(scaled));				
		cur.lr <- RunDataSample(scaled, dose.lr, AdjustDoseCoumagenPCx);
		ttrs.lr <- append(ttrs.lr, cur.lr$meanttr);
		deaths.lr <- append(deaths.lr, cur.lr$death);
		strokes.lr <- append(strokes.lr, cur.lr$stroke);
		ichs.lr <- append(ichs.lr, cur.lr$ich);
		echs.lr <- append(echs.lr, cur.lr$ech);
		
		cat(sprintf("----LR:\n"));
		print(GetPValues(ttrs.lr, strokes.lr, ichs.lr+echs.lr, deaths.lr, ttrs.fixed, strokes.fixed, ichs.fixed+echs.fixed, deaths.fixed), digits=6);

		# private lr
		set.seed(i);
		dose.dplr <- WarfDose(clfun.dplr(scaled));
		cur.dplr <- RunDataSample(scaled, dose.dplr, AdjustDoseCoumagenPCx);
		ttrs.dplr <- append(ttrs.dplr, cur.dplr$meanttr);
		deaths.dplr <- append(deaths.dplr, cur.dplr$death);
		strokes.dplr <- append(strokes.dplr, cur.dplr$stroke);
		ichs.dplr <- append(ichs.dplr, cur.dplr$ich);
		echs.dplr <- append(echs.dplr, cur.dplr$ech);
		
		cat(sprintf("----DPLR:\n"));
		print(GetPValues(ttrs.dplr, strokes.dplr, ichs.dplr+echs.dplr, deaths.dplr, ttrs.fixed, strokes.fixed, ichs.fixed+echs.fixed, deaths.fixed), digits=6);

	}
	
	rawdata <- data.frame(	ttrs.fixed,strokes.fixed,ichs.fixed,echs.fixed,deaths.fixed,
							ttrs.lr,strokes.lr,ichs.lr,echs.lr,deaths.lr,
							ttrs.dplr,strokes.dplr,ichs.dplr,echs.dplr,deaths.dplr);
	pvals <- list(	lr=GetPValues(ttrs.lr, strokes.lr, ichs.lr+echs.lr, deaths.lr, ttrs.fixed, strokes.fixed, ichs.fixed+echs.fixed, deaths.fixed),
					dplr=GetPValues(ttrs.dplr, strokes.dplr, ichs.dplr+echs.dplr, deaths.dplr, ttrs.fixed, strokes.fixed, ichs.fixed+echs.fixed, deaths.fixed));

	curdir <- getwd();
	dir.create(file.path(curdir, "utility-results"), showWarnings = FALSE);
	setwd(file.path(curdir, "utility-results"));
	filename <- sprintf("runComparisonTrials-eps.%.2f-nmodels.%d-%s.Robject", eps, nmodels, format(Sys.time(), "%b%d.%H.%M.%S"));
	dput(list(rawdata=rawdata, pvals=pvals), file=filename);
	setwd(curdir);
	
	return(list(rawdata=rawdata, pvals=pvals));
}


RunComparisonTrialsWithModel <- function(clfun.dplr, training, validation, eps) {
	
	ttrs.fixed <- c();	
	strokes.fixed <- c();
	ichs.fixed <- c();
	echs.fixed <- c();
	deaths.fixed <- c();
	ttrs.lr <- c();	
	strokes.lr <- c();
	ichs.lr <- c();
	echs.lr <- c();
	deaths.lr <- c();
	ttrs.dplr <- c();	
	strokes.dplr <- c();
	ichs.dplr <- c();
	echs.dplr <- c();
	deaths.dplr <- c();
	
	clfun.lr <- diffpLR.reverse.functional(training, "dose", 0, vkoattr)$forward;	

	for(i in 1:nrow(validation)) {				
		
		cat(sprintf("\ni=%d, epsilon=%f\n\n", i, eps));
								
		scaled <- validation[i,];
		
		# fixed-dose
		set.seed(i);
		dose.fixed <- 5.0;
		cur.fixed <- RunDataSample(scaled, dose.fixed, AdjustDoseCoumagen);
		ttrs.fixed <- append(ttrs.fixed, cur.fixed$meanttr);
		deaths.fixed <- append(deaths.fixed, cur.fixed$death);
		strokes.fixed <- append(strokes.fixed, cur.fixed$stroke);
		ichs.fixed <- append(ichs.fixed, cur.fixed$ich);
		echs.fixed <- append(echs.fixed, cur.fixed$ech);
		
		cat(sprintf("----FIXED: ttrs=%.4f strokes=%.4f bleeds=%.4f deaths=%.4f\n", mean(ttrs.fixed), mean(strokes.fixed), mean(ichs.fixed+echs.fixed), mean(deaths.fixed)));	

		# non-private lr
		set.seed(i);
		dose.lr <- WarfDose(clfun.lr(scaled));				
		cur.lr <- RunDataSample(scaled, dose.lr, AdjustDoseCoumagenPCx);
		ttrs.lr <- append(ttrs.lr, cur.lr$meanttr);
		deaths.lr <- append(deaths.lr, cur.lr$death);
		strokes.lr <- append(strokes.lr, cur.lr$stroke);
		ichs.lr <- append(ichs.lr, cur.lr$ich);
		echs.lr <- append(echs.lr, cur.lr$ech);
		
		cat(sprintf("----LR:\n"));
		print(GetPValues(ttrs.lr, strokes.lr, ichs.lr+echs.lr, deaths.lr, ttrs.fixed, strokes.fixed, ichs.fixed+echs.fixed, deaths.fixed), digits=6);

		# private lr
		set.seed(i);
		dose.dplr <- WarfDose(clfun.dplr(scaled));
		cur.dplr <- RunDataSample(scaled, dose.dplr, AdjustDoseCoumagenPCx);
		ttrs.dplr <- append(ttrs.dplr, cur.dplr$meanttr);
		deaths.dplr <- append(deaths.dplr, cur.dplr$death);
		strokes.dplr <- append(strokes.dplr, cur.dplr$stroke);
		ichs.dplr <- append(ichs.dplr, cur.dplr$ich);
		echs.dplr <- append(echs.dplr, cur.dplr$ech);
		
		cat(sprintf("----DPLR:\n"));
		print(GetPValues(ttrs.dplr, strokes.dplr, ichs.dplr+echs.dplr, deaths.dplr, ttrs.fixed, strokes.fixed, ichs.fixed+echs.fixed, deaths.fixed), digits=6);

	}
	
	rawdata <- data.frame(	ttrs.fixed,strokes.fixed,ichs.fixed,echs.fixed,deaths.fixed,
							ttrs.lr,strokes.lr,ichs.lr,echs.lr,deaths.lr,
							ttrs.dplr,strokes.dplr,ichs.dplr,echs.dplr,deaths.dplr);
	pvals <- list(	lr=GetPValues(ttrs.lr, strokes.lr, ichs.lr+echs.lr, deaths.lr, ttrs.fixed, strokes.fixed, ichs.fixed+echs.fixed, deaths.fixed),
					dplr=GetPValues(ttrs.dplr, strokes.dplr, ichs.dplr+echs.dplr, deaths.dplr, ttrs.fixed, strokes.fixed, ichs.fixed+echs.fixed, deaths.fixed));

	curdir <- getwd();
	dir.create(file.path(curdir, "utility-results"), showWarnings = FALSE);
	setwd(file.path(curdir, "utility-results"));
	filename <- sprintf("runComparisonTrials-eps.%.2f-nmodels.%d-%s.Robject", eps, nmodels, format(Sys.time(), "%b%d.%H.%M.%S"));
	dput(list(rawdata=rawdata, pvals=pvals), file=filename);
	setwd(curdir);
	
	return(list(rawdata=rawdata, pvals=pvals));
}

RunComparisonTrialsWithPData <- function(training, validation, fileprefix, nmodels, eps) {
	
	ttrs.fixed <- c();	
	strokes.fixed <- c();
	ichs.fixed <- c();
	echs.fixed <- c();
	deaths.fixed <- c();
	ttrs.lr <- c();	
	strokes.lr <- c();
	ichs.lr <- c();
	echs.lr <- c();
	deaths.lr <- c();
	ttrs.dplr <- c();	
	strokes.dplr <- c();
	ichs.dplr <- c();
	echs.dplr <- c();
	deaths.dplr <- c();
	ttrs.pdata <- c();	
	strokes.pdata <- c();
	ichs.pdata <- c();
	echs.pdata <- c();
	deaths.pdata <- c();
	
	dataset <- 1;

	filename <- sprintf("%s%.2f.2-%d.csv", fileprefix, eps, 1);
	data <- read.csv(filename);
	names(data) <- names(validation);
	data <- as.matrix(data);
	
	clfun.lr <- diffpLR(training, "dose", 0);
	clfun.dplr <- diffpLR(training, "dose", eps);
	clfun.pdata <- diffpLR(data, "dose", 0);

	for(i in 1:nrow(validation)) {				
		
		cat(sprintf("\ni=%d, epsilon=%f\n\n", i, eps));
		
		if(i %% max(1,round(nrow(validation)/nmodels)) == 0) {
			
			dataset <- dataset + 1;
			cat(sprintf("\n(on private version %d)\n\n", dataset))
			
			filename <- paste0(fileprefix, "-", as.character(dataset), ".csv");
			filename <- sprintf("%s%.2f.2-%d.csv", fileprefix, eps, min(dataset, nmodels));
			
			data <- read.csv(filename);
			names(data) <- names(training);
			data <- as.matrix(data);
			
			clfun.dplr <- diffpLR(training, "dose", eps);
			clfun.pdata <- diffpLR(data, "dose", 0);
		}
						
		scaled <- validation[i,];
		
		# fixed-dose
		set.seed(i);
		dose.fixed <- 5.0;
		cur.fixed <- RunDataSample(scaled, dose.fixed, AdjustDoseCoumagen);
		ttrs.fixed <- append(ttrs.fixed, cur.fixed$meanttr);
		deaths.fixed <- append(deaths.fixed, cur.fixed$death);
		strokes.fixed <- append(strokes.fixed, cur.fixed$stroke);
		ichs.fixed <- append(ichs.fixed, cur.fixed$ich);
		echs.fixed <- append(echs.fixed, cur.fixed$ech);
		
		cat(sprintf("----FIXED: ttrs=%.4f strokes=%.4f bleeds=%.4f deaths=%.4f\n", mean(ttrs.fixed), mean(strokes.fixed), mean(ichs.fixed+echs.fixed), mean(deaths.fixed)));	

		# non-private lr
		set.seed(i);
		dose.lr <- WarfDose(clfun.lr(scaled));				
		cur.lr <- RunDataSample(scaled, dose.lr, AdjustDoseCoumagenPCx);
		ttrs.lr <- append(ttrs.lr, cur.lr$meanttr);
		deaths.lr <- append(deaths.lr, cur.lr$death);
		strokes.lr <- append(strokes.lr, cur.lr$stroke);
		ichs.lr <- append(ichs.lr, cur.lr$ich);
		echs.lr <- append(echs.lr, cur.lr$ech);
		
		cat(sprintf("----LR:\n"));
		print(GetPValues(ttrs.lr, strokes.lr, ichs.lr+echs.lr, deaths.lr, ttrs.fixed, strokes.fixed, ichs.fixed+echs.fixed, deaths.fixed), digits=6);

		# private lr
		set.seed(i);
		dose.dplr <- WarfDose(clfun.dplr(scaled));
		cur.dplr <- RunDataSample(scaled, dose.dplr, AdjustDoseCoumagenPCx);
		ttrs.dplr <- append(ttrs.dplr, cur.dplr$meanttr);
		deaths.dplr <- append(deaths.dplr, cur.dplr$death);
		strokes.dplr <- append(strokes.dplr, cur.dplr$stroke);
		ichs.dplr <- append(ichs.dplr, cur.dplr$ich);
		echs.dplr <- append(echs.dplr, cur.dplr$ech);
		
		cat(sprintf("----DPLR:\n"));
		print(GetPValues(ttrs.dplr, strokes.dplr, ichs.dplr+echs.dplr, deaths.dplr, ttrs.fixed, strokes.fixed, ichs.fixed+echs.fixed, deaths.fixed), digits=6);

		# private dataset
		set.seed(i);
		dose.pdata <- WarfDose(clfun.pdata(scaled));				
		cur.pdata <- RunDataSample(scaled, dose.pdata, AdjustDoseCoumagenPCx);
		ttrs.pdata <- append(ttrs.pdata, cur.pdata$meanttr);
		deaths.pdata <- append(deaths.pdata, cur.pdata$death);
		strokes.pdata <- append(strokes.pdata, cur.pdata$stroke);
		ichs.pdata <- append(ichs.pdata, cur.pdata$ich);
		echs.pdata <- append(echs.pdata, cur.pdata$ech);
		
		cat(sprintf("----PDATA:\n"));
		print(GetPValues(ttrs.pdata, strokes.pdata, ichs.pdata+echs.pdata, deaths.pdata, ttrs.fixed, strokes.fixed, ichs.fixed+echs.fixed, deaths.fixed), digits=6);
	}
	
	rawdata <- data.frame(	ttrs.fixed,strokes.fixed,ichs.fixed,echs.fixed,deaths.fixed,
							ttrs.lr,strokes.lr,ichs.lr,echs.lr,deaths.lr,
							ttrs.dplr,strokes.dplr,ichs.dplr,echs.dplr,deaths.dplr,
							ttrs.pdata,strokes.pdata,ichs.pdata,echs.pdata,deaths.pdata);
	pvals <- list(	lr=GetPValues(ttrs.lr, strokes.lr, ichs.lr+echs.lr, deaths.lr, ttrs.fixed, strokes.fixed, ichs.fixed+echs.fixed, deaths.fixed),
					dplr=GetPValues(ttrs.dplr, strokes.dplr, ichs.dplr+echs.dplr, deaths.dplr, ttrs.fixed, strokes.fixed, ichs.fixed+echs.fixed, deaths.fixed),
					pdata=GetPValues(ttrs.pdata, strokes.pdata, ichs.pdata+echs.pdata, deaths.pdata, ttrs.fixed, strokes.fixed, ichs.fixed+echs.fixed, deaths.fixed));

	curdir <- getwd();
	dir.create(file.path(curdir, "utility-results"), showWarnings = FALSE);
	setwd(file.path(curdir, "utility-results"));
	filename <- sprintf("runComparisonTrials-eps.%.2f-nmodels.%d-%s.Robject", eps, nmodels, format(Sys.time(), "%b%d.%H.%M.%S"));
	dput(list(rawdata=rawdata, pvals=pvals), file=filename);
	setwd(curdir);
	
	return(list(rawdata=rawdata, pvals=pvals));
}