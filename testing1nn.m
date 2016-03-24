testimages = loadMNISTImages('t10k-images.idx3-ubyte');
testlabels = loadMNISTLabels('t10k-labels.idx1-ubyte');
trainimages = loadMNISTImages('train-images-idx3-ubyte');
trainlabels = loadMNISTLabels('train-labels-idx1-ubyte');

imgheight=28;
imgwidth=28;
len=100;
noimgs=10000;
i=1;
count=0;
dist=zeros(60000,1);
cm=zeros(10,10);
while(i<=noimgs)
    x=testimages(:,i);
    j=1;
   while(j<=60000)
     y=trainimages(:,j);
     dist(j)=sum((x-y).^2);
     j=j+1;
   end
   [m,I]=min(dist);
   cm(testlabels(i)+1,trainlabels(I)+1)=cm(testlabels(i)+1,trainlabels(I)+1)+1;
   if (testlabels(i)==trainlabels(I))
       count=count+1;
   end
   i=i+1;
end
cm
count/10000*100