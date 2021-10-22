# Generated from esolang_one_byte_opcode_lang.g4 by ANTLR 4.7.2
from antlr4 import *
from io import StringIO
from typing.io import TextIO
import sys


def serializedATN():
    with StringIO() as buf:
        buf.write("\3\u608b\ua72a\u8133\ub9ed\u417c\u3be7\u7786\u5964\2\r")
        buf.write("9\b\1\4\2\t\2\4\3\t\3\4\4\t\4\4\5\t\5\4\6\t\6\4\7\t\7")
        buf.write("\4\b\t\b\4\t\t\t\4\n\t\n\4\13\t\13\4\f\t\f\3\2\3\2\3\3")
        buf.write("\3\3\3\4\3\4\3\5\3\5\3\6\3\6\3\7\3\7\3\b\3\b\3\t\6\t)")
        buf.write("\n\t\r\t\16\t*\3\n\6\n.\n\n\r\n\16\n/\3\13\3\13\6\13\64")
        buf.write("\n\13\r\13\16\13\65\3\f\3\f\2\2\r\3\3\5\4\7\5\t\6\13\7")
        buf.write("\r\b\17\t\21\n\23\13\25\f\27\r\3\2\5\3\2\62;\3\2\63;\5")
        buf.write("\2,-//\61\61\2;\2\3\3\2\2\2\2\5\3\2\2\2\2\7\3\2\2\2\2")
        buf.write("\t\3\2\2\2\2\13\3\2\2\2\2\r\3\2\2\2\2\17\3\2\2\2\2\21")
        buf.write("\3\2\2\2\2\23\3\2\2\2\2\25\3\2\2\2\2\27\3\2\2\2\3\31\3")
        buf.write("\2\2\2\5\33\3\2\2\2\7\35\3\2\2\2\t\37\3\2\2\2\13!\3\2")
        buf.write("\2\2\r#\3\2\2\2\17%\3\2\2\2\21(\3\2\2\2\23-\3\2\2\2\25")
        buf.write("\61\3\2\2\2\27\67\3\2\2\2\31\32\7f\2\2\32\4\3\2\2\2\33")
        buf.write("\34\7h\2\2\34\6\3\2\2\2\35\36\7g\2\2\36\b\3\2\2\2\37 ")
        buf.write("\7x\2\2 \n\3\2\2\2!\"\7c\2\2\"\f\3\2\2\2#$\7i\2\2$\16")
        buf.write("\3\2\2\2%&\7p\2\2&\20\3\2\2\2\')\t\2\2\2(\'\3\2\2\2)*")
        buf.write("\3\2\2\2*(\3\2\2\2*+\3\2\2\2+\22\3\2\2\2,.\t\2\2\2-,\3")
        buf.write("\2\2\2./\3\2\2\2/-\3\2\2\2/\60\3\2\2\2\60\24\3\2\2\2\61")
        buf.write("\63\t\3\2\2\62\64\t\2\2\2\63\62\3\2\2\2\64\65\3\2\2\2")
        buf.write("\65\63\3\2\2\2\65\66\3\2\2\2\66\26\3\2\2\2\678\t\4\2\2")
        buf.write("8\30\3\2\2\2\6\2*/\65\2")
        return buf.getvalue()


class esolang_one_byte_opcode_langLexer(Lexer):

    atn = ATNDeserializer().deserialize(serializedATN())

    decisionsToDFA = [ DFA(ds, i) for i, ds in enumerate(atn.decisionToState) ]

    T__0 = 1
    T__1 = 2
    T__2 = 3
    T__3 = 4
    T__4 = 5
    T__5 = 6
    T__6 = 7
    NUM_DEC = 8
    NUMBER0 = 9
    NUMBER1 = 10
    OPERATOR = 11

    channelNames = [ u"DEFAULT_TOKEN_CHANNEL", u"HIDDEN" ]

    modeNames = [ "DEFAULT_MODE" ]

    literalNames = [ "<INVALID>",
            "'d'", "'f'", "'e'", "'v'", "'a'", "'g'", "'n'" ]

    symbolicNames = [ "<INVALID>",
            "NUM_DEC", "NUMBER0", "NUMBER1", "OPERATOR" ]

    ruleNames = [ "T__0", "T__1", "T__2", "T__3", "T__4", "T__5", "T__6", 
                  "NUM_DEC", "NUMBER0", "NUMBER1", "OPERATOR" ]

    grammarFileName = "esolang_one_byte_opcode_lang.g4"

    def __init__(self, input=None, output:TextIO = sys.stdout):
        super().__init__(input, output)
        self.checkVersion("4.7.2")
        self._interp = LexerATNSimulator(self, self.atn, self.decisionsToDFA, PredictionContextCache())
        self._actions = None
        self._predicates = None


