#! /usr/bin/lua5.3

-- defines a factorial function
function fact (n)
  if n == 0 then
    return 1
  else
    return n * fact(n-1)
  end
end

function fibonacci (n)
    a = 0
    b = 1
    i = 0
    while i < n do
        c = a+b
        a = b
        b = c
        i = i+1
    end

    return b
end
    
print("enter a number:")
a = io.read("*number")        -- read a number
print(fibonacci(a))
-- print(fact(a))
