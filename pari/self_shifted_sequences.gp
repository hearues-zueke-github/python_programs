intLog(a,b)={s=0;while(a>=b,a=a\b;s+=1;);s;};
getExplicitRotatedValue(n,m)={if(n<m,n;,while(n>m-1,b_val=intLog(intLog(n,m),2);mi=m^(2^b_val);n=(n-n\mi)%mi;););n;};
lista(nn)={v=vector(nn);for(n=1,nn,v[n]=getExplicitRotatedValue(n-1,2));v;};
