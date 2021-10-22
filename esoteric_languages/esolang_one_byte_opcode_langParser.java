// Generated from esolang_one_byte_opcode_lang.g4 by ANTLR 4.7.2
import org.antlr.v4.runtime.atn.*;
import org.antlr.v4.runtime.dfa.DFA;
import org.antlr.v4.runtime.*;
import org.antlr.v4.runtime.misc.*;
import org.antlr.v4.runtime.tree.*;
import java.util.List;
import java.util.Iterator;
import java.util.ArrayList;

@SuppressWarnings({"all", "warnings", "unchecked", "unused", "cast"})
public class esolang_one_byte_opcode_langParser extends Parser {
	static { RuntimeMetaData.checkVersion("4.7.2", RuntimeMetaData.VERSION); }

	protected static final DFA[] _decisionToDFA;
	protected static final PredictionContextCache _sharedContextCache =
		new PredictionContextCache();
	public static final int
		T__0=1, T__1=2, T__2=3, T__3=4, T__4=5, T__5=6, NUM_DEC=7, NUMBER0=8, 
		NUMBER1=9, OPERATOR=10;
	public static final int
		RULE_function = 0, RULE_variable = 1, RULE_assign = 2, RULE_number_decimal = 3, 
		RULE_operation = 4;
	private static String[] makeRuleNames() {
		return new String[] {
			"function", "variable", "assign", "number_decimal", "operation"
		};
	}
	public static final String[] ruleNames = makeRuleNames();

	private static String[] makeLiteralNames() {
		return new String[] {
			null, "'f'", "'e'", "'v'", "'a'", "'g'", "'n'"
		};
	}
	private static final String[] _LITERAL_NAMES = makeLiteralNames();
	private static String[] makeSymbolicNames() {
		return new String[] {
			null, null, null, null, null, null, null, "NUM_DEC", "NUMBER0", "NUMBER1", 
			"OPERATOR"
		};
	}
	private static final String[] _SYMBOLIC_NAMES = makeSymbolicNames();
	public static final Vocabulary VOCABULARY = new VocabularyImpl(_LITERAL_NAMES, _SYMBOLIC_NAMES);

	/**
	 * @deprecated Use {@link #VOCABULARY} instead.
	 */
	@Deprecated
	public static final String[] tokenNames;
	static {
		tokenNames = new String[_SYMBOLIC_NAMES.length];
		for (int i = 0; i < tokenNames.length; i++) {
			tokenNames[i] = VOCABULARY.getLiteralName(i);
			if (tokenNames[i] == null) {
				tokenNames[i] = VOCABULARY.getSymbolicName(i);
			}

			if (tokenNames[i] == null) {
				tokenNames[i] = "<INVALID>";
			}
		}
	}

	@Override
	@Deprecated
	public String[] getTokenNames() {
		return tokenNames;
	}

	@Override

	public Vocabulary getVocabulary() {
		return VOCABULARY;
	}

	@Override
	public String getGrammarFileName() { return "esolang_one_byte_opcode_lang.g4"; }

	@Override
	public String[] getRuleNames() { return ruleNames; }

	@Override
	public String getSerializedATN() { return _serializedATN; }

	@Override
	public ATN getATN() { return _ATN; }

	public esolang_one_byte_opcode_langParser(TokenStream input) {
		super(input);
		_interp = new ParserATNSimulator(this,_ATN,_decisionToDFA,_sharedContextCache);
	}
	public static class FunctionContext extends ParserRuleContext {
		public TerminalNode NUMBER0() { return getToken(esolang_one_byte_opcode_langParser.NUMBER0, 0); }
		public TerminalNode EOF() { return getToken(esolang_one_byte_opcode_langParser.EOF, 0); }
		public List<VariableContext> variable() {
			return getRuleContexts(VariableContext.class);
		}
		public VariableContext variable(int i) {
			return getRuleContext(VariableContext.class,i);
		}
		public List<AssignContext> assign() {
			return getRuleContexts(AssignContext.class);
		}
		public AssignContext assign(int i) {
			return getRuleContext(AssignContext.class,i);
		}
		public FunctionContext(ParserRuleContext parent, int invokingState) {
			super(parent, invokingState);
		}
		@Override public int getRuleIndex() { return RULE_function; }
		@Override
		public void enterRule(ParseTreeListener listener) {
			if ( listener instanceof esolang_one_byte_opcode_langListener ) ((esolang_one_byte_opcode_langListener)listener).enterFunction(this);
		}
		@Override
		public void exitRule(ParseTreeListener listener) {
			if ( listener instanceof esolang_one_byte_opcode_langListener ) ((esolang_one_byte_opcode_langListener)listener).exitFunction(this);
		}
	}

	public final FunctionContext function() throws RecognitionException {
		FunctionContext _localctx = new FunctionContext(_ctx, getState());
		enterRule(_localctx, 0, RULE_function);
		int _la;
		try {
			enterOuterAlt(_localctx, 1);
			{
			setState(10);
			match(T__0);
			setState(11);
			match(NUMBER0);
			setState(15);
			_errHandler.sync(this);
			_la = _input.LA(1);
			while (_la==T__2) {
				{
				{
				setState(12);
				variable();
				}
				}
				setState(17);
				_errHandler.sync(this);
				_la = _input.LA(1);
			}
			setState(21);
			_errHandler.sync(this);
			_la = _input.LA(1);
			while (_la==T__4) {
				{
				{
				setState(18);
				assign();
				}
				}
				setState(23);
				_errHandler.sync(this);
				_la = _input.LA(1);
			}
			setState(24);
			match(T__1);
			setState(25);
			match(EOF);
			}
		}
		catch (RecognitionException re) {
			_localctx.exception = re;
			_errHandler.reportError(this, re);
			_errHandler.recover(this, re);
		}
		finally {
			exitRule();
		}
		return _localctx;
	}

	public static class VariableContext extends ParserRuleContext {
		public TerminalNode NUMBER0() { return getToken(esolang_one_byte_opcode_langParser.NUMBER0, 0); }
		public TerminalNode NUMBER1() { return getToken(esolang_one_byte_opcode_langParser.NUMBER1, 0); }
		public VariableContext(ParserRuleContext parent, int invokingState) {
			super(parent, invokingState);
		}
		@Override public int getRuleIndex() { return RULE_variable; }
		@Override
		public void enterRule(ParseTreeListener listener) {
			if ( listener instanceof esolang_one_byte_opcode_langListener ) ((esolang_one_byte_opcode_langListener)listener).enterVariable(this);
		}
		@Override
		public void exitRule(ParseTreeListener listener) {
			if ( listener instanceof esolang_one_byte_opcode_langListener ) ((esolang_one_byte_opcode_langListener)listener).exitVariable(this);
		}
	}

	public final VariableContext variable() throws RecognitionException {
		VariableContext _localctx = new VariableContext(_ctx, getState());
		enterRule(_localctx, 2, RULE_variable);
		try {
			enterOuterAlt(_localctx, 1);
			{
			setState(27);
			match(T__2);
			setState(28);
			match(NUMBER0);
			{
			setState(29);
			match(T__3);
			setState(30);
			match(NUMBER1);
			}
			}
		}
		catch (RecognitionException re) {
			_localctx.exception = re;
			_errHandler.reportError(this, re);
			_errHandler.recover(this, re);
		}
		finally {
			exitRule();
		}
		return _localctx;
	}

	public static class AssignContext extends ParserRuleContext {
		public VariableContext variable() {
			return getRuleContext(VariableContext.class,0);
		}
		public OperationContext operation() {
			return getRuleContext(OperationContext.class,0);
		}
		public AssignContext(ParserRuleContext parent, int invokingState) {
			super(parent, invokingState);
		}
		@Override public int getRuleIndex() { return RULE_assign; }
		@Override
		public void enterRule(ParseTreeListener listener) {
			if ( listener instanceof esolang_one_byte_opcode_langListener ) ((esolang_one_byte_opcode_langListener)listener).enterAssign(this);
		}
		@Override
		public void exitRule(ParseTreeListener listener) {
			if ( listener instanceof esolang_one_byte_opcode_langListener ) ((esolang_one_byte_opcode_langListener)listener).exitAssign(this);
		}
	}

	public final AssignContext assign() throws RecognitionException {
		AssignContext _localctx = new AssignContext(_ctx, getState());
		enterRule(_localctx, 4, RULE_assign);
		try {
			enterOuterAlt(_localctx, 1);
			{
			setState(32);
			match(T__4);
			setState(33);
			variable();
			setState(34);
			operation();
			setState(35);
			match(T__1);
			}
		}
		catch (RecognitionException re) {
			_localctx.exception = re;
			_errHandler.reportError(this, re);
			_errHandler.recover(this, re);
		}
		finally {
			exitRule();
		}
		return _localctx;
	}

	public static class Number_decimalContext extends ParserRuleContext {
		public TerminalNode NUM_DEC() { return getToken(esolang_one_byte_opcode_langParser.NUM_DEC, 0); }
		public Number_decimalContext(ParserRuleContext parent, int invokingState) {
			super(parent, invokingState);
		}
		@Override public int getRuleIndex() { return RULE_number_decimal; }
		@Override
		public void enterRule(ParseTreeListener listener) {
			if ( listener instanceof esolang_one_byte_opcode_langListener ) ((esolang_one_byte_opcode_langListener)listener).enterNumber_decimal(this);
		}
		@Override
		public void exitRule(ParseTreeListener listener) {
			if ( listener instanceof esolang_one_byte_opcode_langListener ) ((esolang_one_byte_opcode_langListener)listener).exitNumber_decimal(this);
		}
	}

	public final Number_decimalContext number_decimal() throws RecognitionException {
		Number_decimalContext _localctx = new Number_decimalContext(_ctx, getState());
		enterRule(_localctx, 6, RULE_number_decimal);
		try {
			enterOuterAlt(_localctx, 1);
			{
			setState(37);
			match(T__5);
			setState(38);
			match(NUM_DEC);
			}
		}
		catch (RecognitionException re) {
			_localctx.exception = re;
			_errHandler.reportError(this, re);
			_errHandler.recover(this, re);
		}
		finally {
			exitRule();
		}
		return _localctx;
	}

	public static class OperationContext extends ParserRuleContext {
		public TerminalNode OPERATOR() { return getToken(esolang_one_byte_opcode_langParser.OPERATOR, 0); }
		public List<VariableContext> variable() {
			return getRuleContexts(VariableContext.class);
		}
		public VariableContext variable(int i) {
			return getRuleContext(VariableContext.class,i);
		}
		public List<OperationContext> operation() {
			return getRuleContexts(OperationContext.class);
		}
		public OperationContext operation(int i) {
			return getRuleContext(OperationContext.class,i);
		}
		public List<Number_decimalContext> number_decimal() {
			return getRuleContexts(Number_decimalContext.class);
		}
		public Number_decimalContext number_decimal(int i) {
			return getRuleContext(Number_decimalContext.class,i);
		}
		public OperationContext(ParserRuleContext parent, int invokingState) {
			super(parent, invokingState);
		}
		@Override public int getRuleIndex() { return RULE_operation; }
		@Override
		public void enterRule(ParseTreeListener listener) {
			if ( listener instanceof esolang_one_byte_opcode_langListener ) ((esolang_one_byte_opcode_langListener)listener).enterOperation(this);
		}
		@Override
		public void exitRule(ParseTreeListener listener) {
			if ( listener instanceof esolang_one_byte_opcode_langListener ) ((esolang_one_byte_opcode_langListener)listener).exitOperation(this);
		}
	}

	public final OperationContext operation() throws RecognitionException {
		OperationContext _localctx = new OperationContext(_ctx, getState());
		enterRule(_localctx, 8, RULE_operation);
		try {
			enterOuterAlt(_localctx, 1);
			{
			setState(40);
			match(OPERATOR);
			setState(44);
			_errHandler.sync(this);
			switch (_input.LA(1)) {
			case T__2:
				{
				setState(41);
				variable();
				}
				break;
			case OPERATOR:
				{
				setState(42);
				operation();
				}
				break;
			case T__5:
				{
				setState(43);
				number_decimal();
				}
				break;
			default:
				throw new NoViableAltException(this);
			}
			setState(49);
			_errHandler.sync(this);
			switch (_input.LA(1)) {
			case T__2:
				{
				setState(46);
				variable();
				}
				break;
			case OPERATOR:
				{
				setState(47);
				operation();
				}
				break;
			case T__5:
				{
				setState(48);
				number_decimal();
				}
				break;
			default:
				throw new NoViableAltException(this);
			}
			setState(51);
			match(T__1);
			}
		}
		catch (RecognitionException re) {
			_localctx.exception = re;
			_errHandler.reportError(this, re);
			_errHandler.recover(this, re);
		}
		finally {
			exitRule();
		}
		return _localctx;
	}

	public static final String _serializedATN =
		"\3\u608b\ua72a\u8133\ub9ed\u417c\u3be7\u7786\u5964\3\f8\4\2\t\2\4\3\t"+
		"\3\4\4\t\4\4\5\t\5\4\6\t\6\3\2\3\2\3\2\7\2\20\n\2\f\2\16\2\23\13\2\3\2"+
		"\7\2\26\n\2\f\2\16\2\31\13\2\3\2\3\2\3\2\3\3\3\3\3\3\3\3\3\3\3\4\3\4\3"+
		"\4\3\4\3\4\3\5\3\5\3\5\3\6\3\6\3\6\3\6\5\6/\n\6\3\6\3\6\3\6\5\6\64\n\6"+
		"\3\6\3\6\3\6\2\2\7\2\4\6\b\n\2\2\28\2\f\3\2\2\2\4\35\3\2\2\2\6\"\3\2\2"+
		"\2\b\'\3\2\2\2\n*\3\2\2\2\f\r\7\3\2\2\r\21\7\n\2\2\16\20\5\4\3\2\17\16"+
		"\3\2\2\2\20\23\3\2\2\2\21\17\3\2\2\2\21\22\3\2\2\2\22\27\3\2\2\2\23\21"+
		"\3\2\2\2\24\26\5\6\4\2\25\24\3\2\2\2\26\31\3\2\2\2\27\25\3\2\2\2\27\30"+
		"\3\2\2\2\30\32\3\2\2\2\31\27\3\2\2\2\32\33\7\4\2\2\33\34\7\2\2\3\34\3"+
		"\3\2\2\2\35\36\7\5\2\2\36\37\7\n\2\2\37 \7\6\2\2 !\7\13\2\2!\5\3\2\2\2"+
		"\"#\7\7\2\2#$\5\4\3\2$%\5\n\6\2%&\7\4\2\2&\7\3\2\2\2\'(\7\b\2\2()\7\t"+
		"\2\2)\t\3\2\2\2*.\7\f\2\2+/\5\4\3\2,/\5\n\6\2-/\5\b\5\2.+\3\2\2\2.,\3"+
		"\2\2\2.-\3\2\2\2/\63\3\2\2\2\60\64\5\4\3\2\61\64\5\n\6\2\62\64\5\b\5\2"+
		"\63\60\3\2\2\2\63\61\3\2\2\2\63\62\3\2\2\2\64\65\3\2\2\2\65\66\7\4\2\2"+
		"\66\13\3\2\2\2\6\21\27.\63";
	public static final ATN _ATN =
		new ATNDeserializer().deserialize(_serializedATN.toCharArray());
	static {
		_decisionToDFA = new DFA[_ATN.getNumberOfDecisions()];
		for (int i = 0; i < _ATN.getNumberOfDecisions(); i++) {
			_decisionToDFA[i] = new DFA(_ATN.getDecisionState(i), i);
		}
	}
}