x = 5       -- x is a number
y = "hello" -- y is a string
z = true    -- z is a boolean


if x > 5 then
  print("x is greater than 5")
else
  print("x is not greater than 5")
end

for i = 1, 10 do
  print(i)
end

while x > 0 do
  x = x - 1
  print(x)
end


function add(x, y)
  return x + y
end

result = add(3, 4)
print(result)  -- prints 7
