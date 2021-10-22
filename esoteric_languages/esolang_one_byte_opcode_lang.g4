grammar esolang_one_byte_opcode_lang;

function : 'f' NUMBER0 variable* assign* 'e' EOF ;
variable : 'v' NUMBER0 ('a' NUMBER1) ;
assign : 'g' variable operation 'e' ;
number_decimal : 'n' NUM_DEC ;
operation : OPERATOR (variable | operation | number_decimal) (variable | operation | number_decimal) 'e' ;

NUM_DEC : [0-9]+ ;

NUMBER0 : [0-9]+ ;
NUMBER1 : [1-9][0-9]+ ;
OPERATOR : '+' | '-' | '*' | '/' ;
