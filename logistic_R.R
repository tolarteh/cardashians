data<-read.csv('data_fill.csv',sep=',',header=TRUE)
attach(data)
logit_model = glm(IsBadBuy ~ PurchDate+Auction+VehYear+VehicleAge+Make+Model+Trim+SubModel+Color+Transmission+WheelTypeID+WheelType+VehOdo+Nationality+Size+TopThreeAmericanName+MMRAcquisitionAuctionAveragePrice+MMRAcquisitionAuctionCleanPrice+MMRAcquisitionRetailAveragePrice+MMRAcquisitonRetailCleanPrice+MMRCurrentAuctionAveragePrice+MMRCurrentAuctionCleanPrice+MMRCurrentRetailAveragePrice+MMRCurrentRetailCleanPrice+PRIMEUNIT+AUCGUART+BYRNO+VNZIP1+VNST+VehBCost+IsOnlineSale+WarrantyCost,family = binomial(link = "logit"))
summary(logit_model)
?step
slm1<-step(logit_model)

