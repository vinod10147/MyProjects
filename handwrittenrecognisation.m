
images = loadMNISTImages('train-images-idx3-ubyte');
labels = loadMNISTLabels('train-labels-idx1-ubyte');

eta=0.5;
imgheight=28;
imgwidth=28;
len=100;
noimgs=600;
imgpixels=imgheight*imgwidth;
w=rand(imgpixels,len)*.09;
v=rand(len,10)*0.09;
i=1;
J=999;
k=1;
format long
xaxis=[];
yaxis=[];
while(J>0.8)
   xaxis(k)=k;
   yaxis(k)=J;
    i=1;
    J=0;
    while(i<=noimgs)
        t(1:10,1)=0.02;
        target=labels(i);
        t(target+1,1)=0.95;
            x=images(:,i);
            netj=(x'*w)';
        y=sigmoid(netj);
        netk=(y'*v)';
        z=sigmoid(netk);
        delk=(t-z).*(z.*(1-z));
        delj=(y).*(1-y).*(v*delk);
        delv=eta*y*delk';
        delw=eta*x*delj';
        w=w+delw;
        v=v+delv;
        J=J+1/2*sum((t-z).^2);
        i=i+1;
    end
    J
    k=k+1;    
end
plot(x,y);