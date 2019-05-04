#include<iostream>
#include<gmpxx.h>

using namespace std;

int main (int argc, char **argv) {
  mpf_t a,b,c;
  mpf_init2(a, 4);
  mpf_init2(b, 5);
  mpf_init2(c, 6);

  mpf_set_str(a, "1234.0001", 10);
  mpf_set_str(b,"-5678", 10); //Decimal base

  mpf_add(c,a,b);

  cout<<"\nThe exact result is:";
  mpf_out_str(stdout, 10, 4, c); //Stream, numerical base, var
  cout<<endl;

  mpf_abs(c, c);
  cout<<"The absolute value result is:";
  // mpf_out_str(stdout, 10, 50, c);
  cout << c;
  cout<<endl;

  cin.get();

  return 0;
}
