from enum import Enum

class Languages(Enum):
    OTHER = 0
    ONES_ENTERPRISE = 1
    ABAP = 2
    ACTIONSCRIPT = 3
    ADA = 4
    APACHE_GROOVY = 5
    APEX = 6
    APPLESCRIPT = 7
    ASP = 8
    ASSEMBLY = 9
    AUTOHOTKEY = 10
    AWK = 11
    BASIC = 12
    BATCH = 13
    BISON = 14
    C = 15
    CLOJURE = 16
    CMAKE = 17
    COBOL = 18
    COFFESCRIPT = 19
    COMMON_LISP = 20
    CPLUSPLUS = 21
    CRYSTAL = 22
    CSHARP = 23
    CSS = 24
    CSV = 25
    D = 26
    DART = 27
    DELPHI = 28
    DOCKER = 29
    ELIXIR = 30
    ELM = 31
    ERLANG = 32
    FIFT = 33
    FORTH = 34
    FORTRAN = 35
    FSHARP = 36
    FUNC = 37
    GAMS = 38
    GO = 39
    GRADLE = 40
    GRAPHQL = 41
    HACK = 42
    HASKELL = 43
    HTML = 44
    ICON = 45
    IDL = 46
    INI = 47
    JAVA = 48
    JAVASCRIPT = 49
    JSON = 50
    JULIA = 51
    KEYMAN = 52
    KOTLIN = 53
    LATEX = 54
    LISP = 55
    LOGO = 56
    LUA = 57
    MAKEFILE = 58
    MARKDOWN = 59
    MATLAB = 60
    NGINX = 61
    NIM = 62
    OBJECTIVE_C = 63
    OCAML = 64
    OPENEDGE_ABL = 65
    PASCAL = 66
    PERL = 67
    PHP = 68
    PL_SQL = 69
    POWERSHELL = 70
    PROLOG = 71
    PROTOBUF = 72
    PYTHON = 73
    QML = 74
    R = 75
    RAKU = 76
    REGEX = 77
    RUBY = 78
    RUST = 79
    SAS = 80
    SCALA = 81
    SCHEME = 82
    SHELL = 83
    SMALLTALK = 84
    SOLIDITY = 85
    SQL = 86
    SWIFT = 87
    TCL = 88
    TEXTILE = 89
    TL = 90
    TYPESCRIPT = 91
    UNREALSCRIPT = 92
    VALA = 93
    VBSCRIPT = 94
    VERILOG = 95
    VISUAL_BASIC = 96
    WOLFRAM = 97
    XML = 98
    YAML = 99

    @staticmethod
    def from_string(s):
        return string_to_enum[s]
    
    @staticmethod
    def to_string(e):
        if isinstance(e, int):
            e = Languages(e)

        return enum_to_string[e]

enum_to_string = {
    Languages.OTHER: "Other",
    Languages.ONES_ENTERPRISE: "1C Enterprise",
    Languages.ABAP: "ABAP",
    Languages.ACTIONSCRIPT: "ActionScript",
    Languages.ADA: "Ada",
    Languages.APACHE_GROOVY: "Groovy",
    Languages.APEX: "APEX",
    Languages.APPLESCRIPT: "AppleScript",
    Languages.ASP: "ASP",
    Languages.ASSEMBLY: "Assembly",
    Languages.AUTOHOTKEY: "AutoHotKey",
    Languages.AWK: "AWK",
    Languages.BASIC: "Basic",
    Languages.BATCH: "Batch",
    Languages.BISON: "Bison",
    Languages.C: "C",
    Languages.CLOJURE: "Clojure",
    Languages.CMAKE: "CMake",
    Languages.COBOL: "COBOL",
    Languages.COFFESCRIPT: "CoffeeScript",
    Languages.COMMON_LISP: "CommonLisp",
    Languages.CPLUSPLUS: "C++",
    Languages.CRYSTAL: "Crystal",
    Languages.CSHARP: "C#",
    Languages.CSS: "CSS",
    Languages.CSV: "CSV",
    Languages.D: "D",
    Languages.DART: "Dart",
    Languages.DELPHI: "Delphi",
    Languages.DOCKER: "Docker",
    Languages.ELIXIR: "Elixir",
    Languages.ELM: "Elm",
    Languages.ERLANG: "Erlang",
    Languages.FIFT: "FIFT",
    Languages.FORTH: "Forth",
    Languages.FORTRAN: "Fortran",
    Languages.FSHARP: "F#",
    Languages.FUNC: "FunC",
    Languages.GAMS: "GAMS",
    Languages.GO: "Go",
    Languages.GRADLE: "Gradle",
    Languages.GRAPHQL: "GraphQL",
    Languages.HACK: "Hack",
    Languages.HASKELL: "Haskell",
    Languages.HTML: "HTML",
    Languages.ICON: "Icon",
    Languages.IDL: "IDL",
    Languages.INI: "INI",
    Languages.JAVA: "Java",
    Languages.JAVASCRIPT: "JavaScript",
    Languages.JSON: "JSON",
    Languages.JULIA: "Julia",
    Languages.KEYMAN: "Keyman",
    Languages.KOTLIN: "Kotlin",
    Languages.LATEX: "LaTeX",
    Languages.LISP: "Lisp",
    Languages.LOGO: "Logo",
    Languages.LUA: "Lua",
    Languages.MAKEFILE: "Makefile",
    Languages.MARKDOWN: "Markdown",
    Languages.MATLAB: "MATLAB",
    Languages.NGINX: "NGINX",
    Languages.NIM: "Nim",
    Languages.OBJECTIVE_C: "Objective-C",
    Languages.OCAML: "OCaml",
    Languages.OPENEDGE_ABL: "OpenEdgeABL",
    Languages.PASCAL: "Pascal",
    Languages.PERL: "Perl",
    Languages.PHP: "PHP",
    Languages.PL_SQL: "PLSQL",
    Languages.POWERSHELL: "PowerShell",
    Languages.PROLOG: "Prolog",
    Languages.PROTOBUF: "Protobuf",
    Languages.PYTHON: "Python",
    Languages.QML: "QML",
    Languages.R: "R",
    Languages.RAKU: "Raku",
    Languages.REGEX: "Regex",
    Languages.RUBY: "Ruby",
    Languages.RUST: "Rust",
    Languages.SAS: "SAS",
    Languages.SCALA: "Scala",
    Languages.SCHEME: "Scheme",
    Languages.SHELL: "Shell",
    Languages.SMALLTALK: "Smalltalk",
    Languages.SOLIDITY: "Solidity",
    Languages.SQL: "SQL",
    Languages.SWIFT: "Swift",
    Languages.TCL: "Tcl",
    Languages.TEXTILE: "Textile",
    Languages.TL: "TL",
    Languages.TYPESCRIPT: "TypeScript",
    Languages.UNREALSCRIPT: "UnrealScript",
    Languages.VALA: "Vala",
    Languages.VBSCRIPT: "VBScript",
    Languages.VERILOG: "Verilog",
    Languages.VISUAL_BASIC: "VisualBasic",
    Languages.WOLFRAM: "Wolfram",
    Languages.XML: "XML",
    Languages.YAML: "YAML",
}

string_to_enum = {s: e for e, s in enum_to_string.items()}
