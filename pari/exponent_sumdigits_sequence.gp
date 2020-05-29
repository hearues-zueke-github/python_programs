a(n)={my(x=0); for(i=0, n-1, x=sumdigits(2^x)); x};
lista(nn)={my(x=0, v=vector(nn+1)); v[1]=0; for(i=1, nn, x=sumdigits(2^x); v[i+1]=x); v};
