# Generated from esolang_one_byte_opcode_lang.g4 by ANTLR 4.7.2
# encoding: utf-8
from antlr4 import *
from io import StringIO
from typing.io import TextIO
import sys

def serializedATN():
    with StringIO() as buf:
        buf.write("\3\u608b\ua72a\u8133\ub9ed\u417c\u3be7\u7786\u5964\3\r")
        buf.write("9\4\2\t\2\4\3\t\3\4\4\t\4\4\5\t\5\4\6\t\6\3\2\3\2\3\2")
        buf.write("\3\2\7\2\21\n\2\f\2\16\2\24\13\2\3\2\7\2\27\n\2\f\2\16")
        buf.write("\2\32\13\2\3\2\3\2\3\2\3\3\3\3\3\3\3\3\3\3\3\4\3\4\3\4")
        buf.write("\3\4\3\4\3\5\3\5\3\5\3\6\3\6\3\6\3\6\5\6\60\n\6\3\6\3")
        buf.write("\6\3\6\5\6\65\n\6\3\6\3\6\3\6\2\2\7\2\4\6\b\n\2\2\29\2")
        buf.write("\f\3\2\2\2\4\36\3\2\2\2\6#\3\2\2\2\b(\3\2\2\2\n+\3\2\2")
        buf.write("\2\f\r\7\3\2\2\r\16\7\4\2\2\16\22\7\13\2\2\17\21\5\4\3")
        buf.write("\2\20\17\3\2\2\2\21\24\3\2\2\2\22\20\3\2\2\2\22\23\3\2")
        buf.write("\2\2\23\30\3\2\2\2\24\22\3\2\2\2\25\27\5\6\4\2\26\25\3")
        buf.write("\2\2\2\27\32\3\2\2\2\30\26\3\2\2\2\30\31\3\2\2\2\31\33")
        buf.write("\3\2\2\2\32\30\3\2\2\2\33\34\7\5\2\2\34\35\7\2\2\3\35")
        buf.write("\3\3\2\2\2\36\37\7\6\2\2\37 \7\13\2\2 !\7\7\2\2!\"\7\f")
        buf.write("\2\2\"\5\3\2\2\2#$\7\b\2\2$%\5\4\3\2%&\5\n\6\2&\'\7\5")
        buf.write("\2\2\'\7\3\2\2\2()\7\t\2\2)*\7\n\2\2*\t\3\2\2\2+/\7\r")
        buf.write("\2\2,\60\5\4\3\2-\60\5\n\6\2.\60\5\b\5\2/,\3\2\2\2/-\3")
        buf.write("\2\2\2/.\3\2\2\2\60\64\3\2\2\2\61\65\5\4\3\2\62\65\5\n")
        buf.write("\6\2\63\65\5\b\5\2\64\61\3\2\2\2\64\62\3\2\2\2\64\63\3")
        buf.write("\2\2\2\65\66\3\2\2\2\66\67\7\5\2\2\67\13\3\2\2\2\6\22")
        buf.write("\30/\64")
        return buf.getvalue()


class esolang_one_byte_opcode_langParser ( Parser ):

    grammarFileName = "esolang_one_byte_opcode_lang.g4"

    atn = ATNDeserializer().deserialize(serializedATN())

    decisionsToDFA = [ DFA(ds, i) for i, ds in enumerate(atn.decisionToState) ]

    sharedContextCache = PredictionContextCache()

    literalNames = [ "<INVALID>", "'d'", "'f'", "'e'", "'v'", "'a'", "'g'", 
                     "'n'" ]

    symbolicNames = [ "<INVALID>", "<INVALID>", "<INVALID>", "<INVALID>", 
                      "<INVALID>", "<INVALID>", "<INVALID>", "<INVALID>", 
                      "NUM_DEC", "NUMBER0", "NUMBER1", "OPERATOR" ]

    RULE_function = 0
    RULE_variable = 1
    RULE_assign = 2
    RULE_number_decimal = 3
    RULE_operation = 4

    ruleNames =  [ "function", "variable", "assign", "number_decimal", "operation" ]

    EOF = Token.EOF
    T__0=1
    T__1=2
    T__2=3
    T__3=4
    T__4=5
    T__5=6
    T__6=7
    NUM_DEC=8
    NUMBER0=9
    NUMBER1=10
    OPERATOR=11

    def __init__(self, input:TokenStream, output:TextIO = sys.stdout):
        print("Working too!")
        super().__init__(input, output)
        self.checkVersion("4.7.2")
        self._interp = ParserATNSimulator(self, self.atn, self.decisionsToDFA, self.sharedContextCache)
        self._predicates = None



    class FunctionContext(ParserRuleContext):

        def __init__(self, parser, parent:ParserRuleContext=None, invokingState:int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser

        def NUMBER0(self):
            return self.getToken(esolang_one_byte_opcode_langParser.NUMBER0, 0)

        def EOF(self):
            return self.getToken(esolang_one_byte_opcode_langParser.EOF, 0)

        def variable(self, i:int=None):
            if i is None:
                return self.getTypedRuleContexts(esolang_one_byte_opcode_langParser.VariableContext)
            else:
                return self.getTypedRuleContext(esolang_one_byte_opcode_langParser.VariableContext,i)


        def assign(self, i:int=None):
            if i is None:
                return self.getTypedRuleContexts(esolang_one_byte_opcode_langParser.AssignContext)
            else:
                return self.getTypedRuleContext(esolang_one_byte_opcode_langParser.AssignContext,i)


        def getRuleIndex(self):
            return esolang_one_byte_opcode_langParser.RULE_function

        def enterRule(self, listener:ParseTreeListener):
            if hasattr( listener, "enterFunction" ):
                listener.enterFunction(self)

        def exitRule(self, listener:ParseTreeListener):
            if hasattr( listener, "exitFunction" ):
                listener.exitFunction(self)




    def function(self):

        localctx = esolang_one_byte_opcode_langParser.FunctionContext(self, self._ctx, self.state)
        self.enterRule(localctx, 0, self.RULE_function)
        self._la = 0 # Token type
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 10
            self.match(esolang_one_byte_opcode_langParser.T__0)
            self.state = 11
            self.match(esolang_one_byte_opcode_langParser.T__1)
            self.state = 12
            self.match(esolang_one_byte_opcode_langParser.NUMBER0)
            self.state = 16
            self._errHandler.sync(self)
            _la = self._input.LA(1)
            while _la==esolang_one_byte_opcode_langParser.T__3:
                self.state = 13
                self.variable()
                self.state = 18
                self._errHandler.sync(self)
                _la = self._input.LA(1)

            self.state = 22
            self._errHandler.sync(self)
            _la = self._input.LA(1)
            while _la==esolang_one_byte_opcode_langParser.T__5:
                self.state = 19
                self.assign()
                self.state = 24
                self._errHandler.sync(self)
                _la = self._input.LA(1)

            self.state = 25
            self.match(esolang_one_byte_opcode_langParser.T__2)
            self.state = 26
            self.match(esolang_one_byte_opcode_langParser.EOF)
        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx

    class VariableContext(ParserRuleContext):

        def __init__(self, parser, parent:ParserRuleContext=None, invokingState:int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser

        def NUMBER0(self):
            return self.getToken(esolang_one_byte_opcode_langParser.NUMBER0, 0)

        def NUMBER1(self):
            return self.getToken(esolang_one_byte_opcode_langParser.NUMBER1, 0)

        def getRuleIndex(self):
            return esolang_one_byte_opcode_langParser.RULE_variable

        def enterRule(self, listener:ParseTreeListener):
            if hasattr( listener, "enterVariable" ):
                listener.enterVariable(self)

        def exitRule(self, listener:ParseTreeListener):
            if hasattr( listener, "exitVariable" ):
                listener.exitVariable(self)




    def variable(self):

        localctx = esolang_one_byte_opcode_langParser.VariableContext(self, self._ctx, self.state)
        self.enterRule(localctx, 2, self.RULE_variable)
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 28
            self.match(esolang_one_byte_opcode_langParser.T__3)
            self.state = 29
            self.match(esolang_one_byte_opcode_langParser.NUMBER0)

            self.state = 30
            self.match(esolang_one_byte_opcode_langParser.T__4)
            self.state = 31
            self.match(esolang_one_byte_opcode_langParser.NUMBER1)
        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx

    class AssignContext(ParserRuleContext):

        def __init__(self, parser, parent:ParserRuleContext=None, invokingState:int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser

        def variable(self):
            return self.getTypedRuleContext(esolang_one_byte_opcode_langParser.VariableContext,0)


        def operation(self):
            return self.getTypedRuleContext(esolang_one_byte_opcode_langParser.OperationContext,0)


        def getRuleIndex(self):
            return esolang_one_byte_opcode_langParser.RULE_assign

        def enterRule(self, listener:ParseTreeListener):
            if hasattr( listener, "enterAssign" ):
                listener.enterAssign(self)

        def exitRule(self, listener:ParseTreeListener):
            if hasattr( listener, "exitAssign" ):
                listener.exitAssign(self)




    def assign(self):

        localctx = esolang_one_byte_opcode_langParser.AssignContext(self, self._ctx, self.state)
        self.enterRule(localctx, 4, self.RULE_assign)
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 33
            self.match(esolang_one_byte_opcode_langParser.T__5)
            self.state = 34
            self.variable()
            self.state = 35
            self.operation()
            self.state = 36
            self.match(esolang_one_byte_opcode_langParser.T__2)
        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx

    class Number_decimalContext(ParserRuleContext):

        def __init__(self, parser, parent:ParserRuleContext=None, invokingState:int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser

        def NUM_DEC(self):
            return self.getToken(esolang_one_byte_opcode_langParser.NUM_DEC, 0)

        def getRuleIndex(self):
            return esolang_one_byte_opcode_langParser.RULE_number_decimal

        def enterRule(self, listener:ParseTreeListener):
            if hasattr( listener, "enterNumber_decimal" ):
                listener.enterNumber_decimal(self)

        def exitRule(self, listener:ParseTreeListener):
            if hasattr( listener, "exitNumber_decimal" ):
                listener.exitNumber_decimal(self)




    def number_decimal(self):

        localctx = esolang_one_byte_opcode_langParser.Number_decimalContext(self, self._ctx, self.state)
        self.enterRule(localctx, 6, self.RULE_number_decimal)
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 38
            self.match(esolang_one_byte_opcode_langParser.T__6)
            self.state = 39
            self.match(esolang_one_byte_opcode_langParser.NUM_DEC)
        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx

    class OperationContext(ParserRuleContext):

        def __init__(self, parser, parent:ParserRuleContext=None, invokingState:int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser

        def OPERATOR(self):
            return self.getToken(esolang_one_byte_opcode_langParser.OPERATOR, 0)

        def variable(self, i:int=None):
            if i is None:
                return self.getTypedRuleContexts(esolang_one_byte_opcode_langParser.VariableContext)
            else:
                return self.getTypedRuleContext(esolang_one_byte_opcode_langParser.VariableContext,i)


        def operation(self, i:int=None):
            if i is None:
                return self.getTypedRuleContexts(esolang_one_byte_opcode_langParser.OperationContext)
            else:
                return self.getTypedRuleContext(esolang_one_byte_opcode_langParser.OperationContext,i)


        def number_decimal(self, i:int=None):
            if i is None:
                return self.getTypedRuleContexts(esolang_one_byte_opcode_langParser.Number_decimalContext)
            else:
                return self.getTypedRuleContext(esolang_one_byte_opcode_langParser.Number_decimalContext,i)


        def getRuleIndex(self):
            return esolang_one_byte_opcode_langParser.RULE_operation

        def enterRule(self, listener:ParseTreeListener):
            if hasattr( listener, "enterOperation" ):
                listener.enterOperation(self)

        def exitRule(self, listener:ParseTreeListener):
            if hasattr( listener, "exitOperation" ):
                listener.exitOperation(self)




    def operation(self):

        localctx = esolang_one_byte_opcode_langParser.OperationContext(self, self._ctx, self.state)
        self.enterRule(localctx, 8, self.RULE_operation)
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 41
            self.match(esolang_one_byte_opcode_langParser.OPERATOR)
            self.state = 45
            self._errHandler.sync(self)
            token = self._input.LA(1)
            if token in [esolang_one_byte_opcode_langParser.T__3]:
                self.state = 42
                self.variable()
                pass
            elif token in [esolang_one_byte_opcode_langParser.OPERATOR]:
                self.state = 43
                self.operation()
                pass
            elif token in [esolang_one_byte_opcode_langParser.T__6]:
                self.state = 44
                self.number_decimal()
                pass
            else:
                raise NoViableAltException(self)

            self.state = 50
            self._errHandler.sync(self)
            token = self._input.LA(1)
            if token in [esolang_one_byte_opcode_langParser.T__3]:
                self.state = 47
                self.variable()
                pass
            elif token in [esolang_one_byte_opcode_langParser.OPERATOR]:
                self.state = 48
                self.operation()
                pass
            elif token in [esolang_one_byte_opcode_langParser.T__6]:
                self.state = 49
                self.number_decimal()
                pass
            else:
                raise NoViableAltException(self)

            self.state = 52
            self.match(esolang_one_byte_opcode_langParser.T__2)
        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx





