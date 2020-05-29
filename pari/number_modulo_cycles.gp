checkIfUnique(v,n)={v2=vector(n);unique=1;for(i=1,n,j=v[i]+1;if(v2[j]==1,unique=0;break,v2[j]=1;););unique;};
doCycle(a,c,m)={v_=vector(m);x=c;v_[1]=c;for(i=1,m-1,v_[i+1]=(a*v_[i]+c)%m;);v_;};
getCycles(m)={M=matrix(0,m);for(a=0,m-1,for(c=0,m-1,v1=doCycle(a,c,m);if(checkIfUnique(v1,m),M=matconcat([M;v1]););););M;};
lista(nn)={v=vector(nn);for(i=1,nn,M=getCycles(i);v[i]=matsize(M)[1];);v;};
v=lista(50);
print(Str("v: ",v))
/*quit;*/
