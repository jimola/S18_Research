to_num_fac <- function(row){
    if(class(row) == 'integer')
        return(as.numeric(row))
    else if(class(row) == 'character')
        return(as.factor(row))
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

