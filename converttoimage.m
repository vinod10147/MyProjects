function A=converttoimage(B)
    A=zeros(28,28);
    k=1;
   
    for i=1:28
        for j=1:28
            A(j,i)=B(k);
            k=k+1;
        end
    end
    
end