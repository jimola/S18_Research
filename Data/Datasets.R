to_num_fac <- function(row){
    if(class(row) == 'character')
        return(as.factor(row))
    else if(class(row) == 'integer' || class(row) == 'numeric'){
        L <- unique(row) %>% length
        if(L > 30)
            return(as.numeric(row))
        else
            return(as.factor(row))
    }
    return(row)
}

#Import adult dataset
adult <- fread('../data-unsorted/adult/adult_cleaner_no_relationship.csv')
adult <- lapply(adult, to_num_fac) %>% data.frame
#We need to know how to compute maxes
adult_maxes = list(age=list(min=16, max=100), 
                   education.num=list(min=1, max=16), 
                   capital.gain=list(min=0, max=99999),
                   capital.loss=list(min=0, max=99999),
                   hours.per.week=list(min=1, max=100)
                  )
adult_x_names <- names(adult)[-13]
adult_y_names <- names(adult)[13]
adult_lvls <- levels(adult$earning)
adultd <- list(data=adult, x_names=adult_x_names, y_names=adult_y_names, rng=adult_maxes)

india <- fread('../data-unsorted/india/india-processed.csv')
india <- lapply(india, to_num_fac) %>% data.frame
india$JobCity <- NULL
india$Designation <- NULL
india$X10board <- NULL
india$X12board <- NULL
india$CollegeID <- NULL
india$CollegeCityID <- NULL
india$Domain[india$Domain == -1] = NA
india$ComputerProgramming[india$ComputerProgramming == -1] = NA
india$ElectronicsAndSemicon[india$ElectronicsAndSemicon == -1] = NA
india$ComputerScience[india$ComputerScience == -1] = NA

#india_maxes <- list(X10percentage=list(min=0, max=100) X12percentage=list(min=0, max=100)
#                    Specialization=list(min=, max=), collegeGPA=list(min=0, max=100),
#                    English=list(min=100, max=900), Logical=list(min=100, max=900),
#                    Quant=list(min=100, max=900), Domain=list(min=0, max=1),
#                    ComputerProgramming=list(min=100, max=900),
#                    ElectronicsAndSemicon=list(min=100, max=900),
#                    MechanicalEngg=list(min=, max=),
#                    ElectricalEngg=list(min=, max=), TelecomEngg=list(min=, max=),
#                    CivilEngg=list(min=, max=), conscientiousness=list(min=, max=),
#                    agreeableness=list(min=, max=), extraversion=list(min=, max=),
#                    neuroticism=list(min=, max=), openess_to_experience=list(min=, max=)
#                   )


bind <- fread('../datasets/1625Data.txt')
b1 <- sapply(1:8, function(r) sapply(bind$V1, function(x) substring(x, r, r))) %>% data.frame
bind <- cbind(b1, V2=bind$V2 %>% as.factor)
bind_x_names <- names(bind)[-9]
bind_y_names <- names(bind)[9]
bind_lvls <- levels(bind$V2)
bindd <- list(data=bind, x_names=bind_x_names, y_names=bind_y_names, rng=NA)

ttt <- fread('../datasets/tic-tac-toe.data', header=FALSE)
ttt <- lapply(ttt, to_num_fac) %>% data.frame
ttt <- ttt[sample(1:nrow(ttt), nrow(ttt)), ]
ttt <- list(data=ttt, x_names=names(ttt)[-10], y_names=names(ttt)[10], rng=NA)

nurs <- fread('../datasets/nursery.data') %>% lapply(to_num_fac) %>% data.frame
nurs <- list(data=nurs, x_names=names(nurs)[-9], y_names=names(nurs)[9], rng=NA)

