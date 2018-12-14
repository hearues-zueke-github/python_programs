#include <gmp.h>
 
int main(void)
{
  mpz_t n;
  mpz_init_set_ui(n, 100UL);
  mpz_pow_ui(n, n, 100UL);
  gmp_printf("result: %Zd\n", n);

  return 0;
}
