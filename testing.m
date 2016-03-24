images = loadMNISTImages('t10k-images.idx3-ubyte');
labels = loadMNISTLabels('t10k-labels.idx1-ubyte');
imgheight=28;
imgwidth=28;
len=100;
noimgs=10000;
i=1;
count=0;
cm=zeros(10,10);
w=xlsread('wnew');
v=xlsread('vnew');
mat=zeros(2,2,10);
while(i<=noimgs)
    x=images(:,i);
    netj=(x'*w)';
    y=sigmoid(netj);
    netk=(y'*v)';
    z=sigmoid(netk);
    [m,j]=max(z);
    t=labels(i);
    cm(t+1,j)=cm(t+1,j)+1;
    if(t==j-1)
        count=count+1;
        mat(1,1,j)=mat(1,1,j)+1;
    end
    for p=1:10
       	if p == labels(i) && p~=j-1
       		mat(1,2,p)=mat(1,2,p)+1;
       	elseif p~=labels(i) && p==j-1
       		mat(2,1,p)=mat(2,1,p)+1;
       	elseif  p~=labels(i) && p~=j-1
       		mat(2,2,p)=mat(2,2,p)+1;
        end
    end
    i=i+1;
end
display('Confusion matix');
cm
mat
prec=0;
sens=0;
spec=0;
for p=1:10
    prec=prec+mat(1,1,p)/(mat(1,1,p)+mat(1,2,p));
    sens=sens+mat(1,1,p)/(mat(1,1,p)+mat(2,1,p));
    spec=spec+mat(2,2,p)/(mat(2,2,p)+mat(1,2,p));
    
end
display('Precision')
prec/10

display('Sensitivity')
sens/10

display('Specificity')
spec/10

display('Accuracy');
count/10000*100