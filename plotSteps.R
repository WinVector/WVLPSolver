library(plyr)
library(reshape)
library(ggplot2)
d <- read.table('assignmentSpeed.tsv',header=T,sep='\t')
d2 <- melt(d,id.vars=c('assignmentSize'))
d2 <- subset(d2,d2$variable %in% list('assignmentSize',
  'ApacheM3Simplex','GLPK','WVLPSolver'))
ggplot(data=d2,aes(x=assignmentSize,y=value,color=variable)) + geom_point(alpha=0.5) + geom_smooth(alpha=0.7)
ggsave('assignmentSpeed.png',width=8,height=6,dpi=90)
d2$value = pmax(1,d2$value)
ggplot(data=d2,aes(x=assignmentSize,y=value,color=variable)) + geom_point(alpha=0.5) + geom_smooth(alpha=0.7) +  scale_y_log10() + scale_x_log10()
ggsave('assignmentLogSpeed.png',width=8,height=6,dpi=90)
summary(lm(log(value) ~ 0 + variable + variable:log(assignmentSize),data=subset(d2,d2$assignmentSize>40)))




