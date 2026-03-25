/*
 * chat.m — Interactive TUI chat client for Flash-MoE inference server
 *
 * Thin HTTP/SSE client with session persistence.
 * Conversations saved to ~/.flash-moe/sessions/<session_id>.jsonl
 * Resume with: ./chat --resume <session_id>
 *
 * Build:  make chat
 * Run:    ./chat [--port 8000] [--show-think] [--resume <id>]
 */

#import <Foundation/Foundation.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <sys/time.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <getopt.h>
#include <dirent.h>
#include "linenoise.h"

#define MAX_INPUT_LINE 4096
#define MAX_RESPONSE (1024 * 1024)
#define SESSIONS_DIR_BASE ".flash-moe/sessions"

static double now_ms(void) {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec * 1000.0 + tv.tv_usec / 1000.0;
}

static int json_escape(const char *src, char *buf, int bufsize) {
    int j = 0;
    for (int i = 0; src[i] && j < bufsize - 6; i++) {
        switch (src[i]) {
            case '"':  buf[j++]='\\'; buf[j++]='"'; break;
            case '\\': buf[j++]='\\'; buf[j++]='\\'; break;
            case '\n': buf[j++]='\\'; buf[j++]='n'; break;
            case '\r': buf[j++]='\\'; buf[j++]='r'; break;
            case '\t': buf[j++]='\\'; buf[j++]='t'; break;
            default:   buf[j++]=src[i]; break;
        }
    }
    buf[j] = 0;
    return j;
}

// ============================================================================
// Session persistence
// ============================================================================

static char g_sessions_dir[1024];

static void init_sessions_dir(void) {
    const char *home = getenv("HOME");
    if (!home) home = "/tmp";
    snprintf(g_sessions_dir, sizeof(g_sessions_dir), "%s/%s", home, SESSIONS_DIR_BASE);
    mkdir(g_sessions_dir, 0755);
    // Also create parent
    char parent[1024];
    snprintf(parent, sizeof(parent), "%s/.flash-moe", home);
    mkdir(parent, 0755);
    mkdir(g_sessions_dir, 0755);
}

static void session_path(const char *session_id, char *path, size_t pathsize) {
    snprintf(path, pathsize, "%s/%s.jsonl", g_sessions_dir, session_id);
}

// Append a turn to the session JSONL file
static void session_save_turn(const char *session_id, const char *role, const char *content) {
    char path[1024];
    session_path(session_id, path, sizeof(path));
    FILE *f = fopen(path, "a");
    if (!f) return;
    char escaped[MAX_RESPONSE * 2];
    json_escape(content, escaped, sizeof(escaped));
    fprintf(f, "{\"role\":\"%s\",\"content\":\"%s\"}\n", role, escaped);
    fclose(f);
}

// Load session history and replay to screen
static int session_load(const char *session_id) {
    char path[1024];
    session_path(session_id, path, sizeof(path));
    FILE *f = fopen(path, "r");
    if (!f) return 0;

    printf("[resuming session %s]\n\n", session_id);
    int turns = 0;
    char line[MAX_RESPONSE];
    while (fgets(line, sizeof(line), f)) {
        // Simple parsing: find role and content
        char *role_start = strstr(line, "\"role\":\"");
        char *content_start = strstr(line, "\"content\":\"");
        if (!role_start || !content_start) continue;

        role_start += 8;
        char role[32]; int ri = 0;
        while (*role_start && *role_start != '"' && ri < 31) role[ri++] = *role_start++;
        role[ri] = 0;

        content_start += 11;
        // Decode the content (unescape)
        char content[MAX_RESPONSE]; int ci = 0;
        for (int i = 0; content_start[i] && ci < MAX_RESPONSE - 1; i++) {
            // Stop at closing quote (not escaped)
            if (content_start[i] == '"' && (i == 0 || content_start[i-1] != '\\')) break;
            if (content_start[i] == '\\' && content_start[i+1]) {
                i++;
                switch (content_start[i]) {
                    case 'n': content[ci++] = '\n'; break;
                    case 't': content[ci++] = '\t'; break;
                    case '"': content[ci++] = '"'; break;
                    case '\\': content[ci++] = '\\'; break;
                    default: content[ci++] = content_start[i]; break;
                }
            } else {
                content[ci++] = content_start[i];
            }
        }
        content[ci] = 0;

        if (strcmp(role, "user") == 0) {
            printf("\033[1m> %s\033[0m\n\n", content);
        } else if (strcmp(role, "assistant") == 0) {
            printf("%s\n\n", content);
        }
        turns++;
    }
    fclose(f);
    if (turns > 0) printf("[%d turns loaded]\n\n", turns);
    return turns;
}

// List recent sessions
static void session_list(void) {
    DIR *dir = opendir(g_sessions_dir);
    if (!dir) { printf("No sessions found.\n\n"); return; }

    printf("Recent sessions:\n");
    struct dirent *entry;
    int count = 0;
    while ((entry = readdir(dir))) {
        if (entry->d_name[0] == '.') continue;
        char *dot = strrchr(entry->d_name, '.');
        if (!dot || strcmp(dot, ".jsonl") != 0) continue;
        *dot = 0; // strip .jsonl

        char path[1024];
        snprintf(path, sizeof(path), "%s/%s.jsonl", g_sessions_dir, entry->d_name);
        struct stat st;
        stat(path, &st);

        // Count lines (turns)
        FILE *f = fopen(path, "r");
        int lines = 0;
        if (f) {
            char buf[1024];
            while (fgets(buf, sizeof(buf), f)) lines++;
            fclose(f);
        }

        printf("  %s  (%d turns)\n", entry->d_name, lines);
        count++;
    }
    closedir(dir);
    if (count == 0) printf("  (none)\n");
    printf("\n");
}

// ============================================================================
// HTTP / SSE
// ============================================================================

static void generate_session_id(char *buf, size_t bufsize) {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    snprintf(buf, bufsize, "chat-%d-%ld%06d",
             (int)getpid(), (long)tv.tv_sec, (int)tv.tv_usec);
}

static int send_chat_request(int port, const char *user_message, int max_tokens, const char *session_id) {
    int sock = socket(AF_INET, SOCK_STREAM, 0);
    if (sock < 0) { perror("socket"); return -1; }

    struct sockaddr_in addr = {0};
    addr.sin_family = AF_INET;
    addr.sin_port = htons(port);
    addr.sin_addr.s_addr = inet_addr("127.0.0.1");

    if (connect(sock, (struct sockaddr *)&addr, sizeof(addr)) < 0) {
        fprintf(stderr, "\n[error] Cannot connect to server on port %d.\n", port);
        close(sock);
        return -1;
    }

    char escaped[MAX_INPUT_LINE * 2];
    json_escape(user_message, escaped, sizeof(escaped));

    char body[MAX_INPUT_LINE * 3];
    int body_len = snprintf(body, sizeof(body),
        "{\"messages\":[{\"role\":\"user\",\"content\":\"%s\"}],"
        "\"max_tokens\":%d,\"stream\":true,\"session_id\":\"%s\"}",
        escaped, max_tokens, session_id);

    char request[MAX_INPUT_LINE * 4];
    int req_len = snprintf(request, sizeof(request),
        "POST /v1/chat/completions HTTP/1.1\r\n"
        "Host: localhost:%d\r\n"
        "Content-Type: application/json\r\n"
        "Content-Length: %d\r\n"
        "Connection: close\r\n"
        "\r\n"
        "%s",
        port, body_len, body);

    write(sock, request, req_len);
    return sock;
}

// ============================================================================
// Streaming markdown renderer — stateful ANSI escape code emitter
// ============================================================================
// Handles: **bold**, *italic*, `inline code`, ```code blocks```, # headers
// State persists across token boundaries (e.g. "**" in one token, text in next)

#define ANSI_RESET   "\033[0m"
#define ANSI_BOLD    "\033[1m"
#define ANSI_ITALIC  "\033[3m"
#define ANSI_CODE    "\033[36m"      // cyan for inline code
#define ANSI_CODEBLK "\033[48;5;236m\033[38;5;252m"  // dark bg + light fg
#define ANSI_CODEBLK_LINE "\033[48;5;236m\033[K"     // extend bg to end of line
#define ANSI_HEADER  "\033[1;34m"    // bold blue for headers
#define ANSI_DIM     "\033[2m"

typedef struct {
    int bold;        // inside **...**
    int italic;      // inside *...*
    int code_inline; // inside `...`
    int code_block;  // inside ```...```
    int skip_lang;   // eating language tag after opening ```
    int line_start;  // at start of a new line
    char pending[8]; // buffered chars for lookahead (e.g., partial "**")
    int pending_len;
} MdState;

static MdState g_md = {0, 0, 0, 0, 1, {0}, 0};

static void md_reset(void) {
    memset(&g_md, 0, sizeof(g_md));
    g_md.line_start = 1;
}

static void md_print(const char *text) {
    for (int i = 0; text[i]; i++) {
        char c = text[i];

        // Skip language tag after opening ``` (may span tokens)
        if (g_md.skip_lang) {
            if (c == '\n') {
                g_md.skip_lang = 0;
                printf(ANSI_CODEBLK ANSI_CODEBLK_LINE "\n");
            }
            // else: eat the character (language tag)
            continue;
        }

        // Code block toggle: ```
        if (c == '`' && text[i+1] == '`' && text[i+2] == '`') {
            if (g_md.code_block) {
                printf(ANSI_RESET "\n");
                g_md.code_block = 0;
            } else {
                g_md.code_block = 1;
                g_md.skip_lang = 1;  // eat language tag until newline
            }
            i += 2;
            continue;
        }

        // Inside code block: print with full-width background
        if (g_md.code_block) {
            printf(ANSI_CODEBLK);
            if (c == '\n') {
                printf(ANSI_CODEBLK_LINE "\n");
            } else {
                putchar(c);
            }
            continue;
        }

        // Inline code toggle: `
        if (c == '`') {
            if (g_md.code_inline) {
                printf(ANSI_RESET);
                g_md.code_inline = 0;
            } else {
                printf(ANSI_CODE);
                g_md.code_inline = 1;
            }
            continue;
        }

        // Inside inline code: print verbatim
        if (g_md.code_inline) {
            putchar(c);
            continue;
        }

        // Headers at line start: # ## ### — hide markers, show text bold blue
        if (g_md.line_start && c == '#') {
            while (text[i] == '#') i++;  // skip all #
            while (text[i] == ' ') i++;  // skip space after #
            printf(ANSI_HEADER);
            while (text[i] && text[i] != '\n') { putchar(text[i]); i++; }
            printf(ANSI_RESET);
            if (text[i] == '\n') { putchar('\n'); g_md.line_start = 1; }
            continue;
        }

        // Bullet lists: - or * at line start (possibly indented with spaces)
        // Count leading spaces for indent level, then check for bullet marker
        if (g_md.line_start && (c == '-' || c == '*' || c == ' ')) {
            // Peek ahead: count indent, find marker
            int indent = 0;
            int peek = i;
            while (text[peek] == ' ' || text[peek] == '\t') { indent++; peek++; }
            char marker = text[peek];
            if ((marker == '-' || marker == '*') && marker != '\0') {
                char after = text[peek + 1];
                // Bullet: marker followed by space, end of token, or tab
                // For *, must not be ** (bold)
                if (marker == '-' && (after == ' ' || after == '\0')) {
                    int depth = indent / 2;
                    for (int d = 0; d < depth + 1; d++) printf("  ");
                    printf("\033[33m•\033[0m ");
                    i = peek + 1;
                    while (text[i] == ' ' || text[i] == '\t') i++;
                    i--; // loop will i++
                    g_md.line_start = 0;
                    continue;
                }
                if (marker == '*' && after != '*' && (after == ' ' || after == '\0' || after == '\t')) {
                    int depth = indent / 2;
                    for (int d = 0; d < depth + 1; d++) printf("  ");
                    printf("\033[33m•\033[0m ");
                    i = peek + 1;
                    while (text[i] == ' ' || text[i] == '\t') i++;
                    i--;
                    g_md.line_start = 0;
                    continue;
                }
            }
            // Not a bullet — fall through to normal handling
        }

        // Numbered lists at line start: 1. item → colored number
        if (g_md.line_start && c >= '0' && c <= '9') {
            int num_start = i;
            while (text[i] >= '0' && text[i] <= '9') i++;
            if (text[i] == '.' && text[i+1] == ' ') {
                printf("  \033[33m");  // yellow
                for (int j = num_start; j <= i; j++) putchar(text[j]);
                printf("\033[0m");
                i++; // skip space
                g_md.line_start = 0;
                continue;
            }
            // Not a list, rewind and print normally
            i = num_start;
            c = text[i];
        }

        // Bold: **
        if (c == '*' && text[i+1] == '*') {
            if (g_md.bold) {
                printf(ANSI_RESET);
                g_md.bold = 0;
            } else {
                printf(ANSI_BOLD);
                g_md.bold = 1;
            }
            i++;
            continue;
        }

        // Italic: single * (but not **)
        if (c == '*' && text[i+1] != '*') {
            if (g_md.italic) {
                printf(ANSI_RESET);
                g_md.italic = 0;
            } else {
                printf(ANSI_ITALIC);
                g_md.italic = 1;
            }
            continue;
        }

        // Track line starts
        if (c == '\n') {
            g_md.line_start = 1;
        } else {
            g_md.line_start = 0;
        }

        putchar(c);
    }
}

// Stream SSE response, accumulate text, return malloc'd response string
static char *stream_response(int sock, int show_thinking) {
    FILE *stream = fdopen(sock, "r");
    if (!stream) { close(sock); return NULL; }

    int header_done = 0, in_think = 0, tokens = 0;
    double t_start = now_ms(), t_first = 0;
    md_reset();  // fresh markdown state for each response

    char *response = calloc(1, MAX_RESPONSE);
    int resp_len = 0;

    char line[65536];
    while (fgets(line, sizeof(line), stream)) {
        if (!header_done) {
            if (strcmp(line, "\r\n") == 0 || strcmp(line, "\n") == 0) header_done = 1;
            continue;
        }
        if (strncmp(line, "data: ", 6) != 0) continue;
        if (strncmp(line + 6, "[DONE]", 6) == 0) break;

        char *ck = strstr(line + 6, "\"content\":\"");
        if (!ck) continue;
        ck += 11;

        char decoded[4096]; int di = 0;
        for (int i = 0; ck[i] && ck[i] != '"' && di < 4095; i++) {
            if (ck[i] == '\\' && ck[i+1]) {
                i++;
                switch (ck[i]) {
                    case 'n': decoded[di++]='\n'; break;
                    case 't': decoded[di++]='\t'; break;
                    case '"': decoded[di++]='"'; break;
                    case '\\': decoded[di++]='\\'; break;
                    default: decoded[di++]=ck[i]; break;
                }
            } else decoded[di++] = ck[i];
        }
        decoded[di] = 0;
        if (!di) continue;

        if (strstr(decoded, "<think>")) in_think = 1;
        if (strstr(decoded, "</think>")) { in_think = 0; tokens++; continue; }
        tokens++;
        if (!t_first) t_first = now_ms();

        // Accumulate non-thinking response
        if (!in_think && resp_len + di < MAX_RESPONSE - 1) {
            memcpy(response + resp_len, decoded, di);
            resp_len += di;
            response[resp_len] = 0;
        }

        if (in_think && !show_thinking) continue;
        if (in_think) printf(ANSI_DIM "%s" ANSI_RESET, decoded);
        else md_print(decoded);
        fflush(stdout);
    }
    fclose(stream);

    printf(ANSI_RESET);  // ensure no style leaks
    double gen_time = t_first > 0 ? now_ms() - t_first : 0;
    int gen_tokens = tokens > 1 ? tokens - 1 : 0;
    printf("\n\n");
    if (gen_tokens > 0 && gen_time > 0)
        printf("[%d tokens, %.1f tok/s, TTFT %.1fs]\n\n",
               tokens, gen_tokens * 1000.0 / gen_time,
               t_first > 0 ? (t_first - now_ms() + gen_time + (t_first - (now_ms() - gen_time))) / 1000.0 : 0);

    return response;
}

// ============================================================================
// Main
// ============================================================================

int main(int argc, char **argv) {
    int port = 8000;
    int max_tokens = 8192;
    int show_thinking = 0;
    const char *resume_id = NULL;

    static struct option long_options[] = {
        {"port",        required_argument, 0, 'p'},
        {"max-tokens",  required_argument, 0, 't'},
        {"show-think",  no_argument,       0, 's'},
        {"resume",      required_argument, 0, 'r'},
        {"sessions",    no_argument,       0, 'l'},
        {"prefill-skip-experts",       no_argument, 0, 0x104},
        {"prefill-experts-full-only",  no_argument, 0, 0x105},
        {"help",        no_argument,       0, 'h'},
        {0, 0, 0, 0}
    };

    init_sessions_dir();

    int c;
    while ((c = getopt_long(argc, argv, "p:t:sr:lh", long_options, NULL)) != -1) {
        switch (c) {
            case 'p': port = atoi(optarg); break;
            case 't': max_tokens = atoi(optarg); break;
            case 's': show_thinking = 1; break;
            case 'r': resume_id = optarg; break;
            case 'l': session_list(); return 0;
            case 0x104: g_prefill_skip_experts = 1; break;
            case 0x105: g_prefill_experts_full_only = 1; break;
            case 'h':
                printf("Usage: %s [options]\n", argv[0]);
                printf("  --port N         Server port (default: 8000)\n");
                printf("  --max-tokens N   Max response tokens (default: 8192)\n");
                printf("  --show-think     Show <think> blocks (dimmed)\n");
                printf("  --resume ID      Resume a previous session\n");
                printf("  --sessions       List saved sessions\n");
                printf("  --prefill-skip-experts    Skip routed experts during prefill (shared only)\n");
                printf("  --prefill-experts-full-only  Experts only at full attention layers during prefill\n");
                printf("  --help           This message\n");
                return 0;
            default: return 1;
        }
    }

    char session_id[64];
    if (resume_id) {
        strncpy(session_id, resume_id, sizeof(session_id) - 1);
        session_id[sizeof(session_id) - 1] = 0;
    } else {
        generate_session_id(session_id, sizeof(session_id));
    }

    printf("==================================================\n");
    printf("  Qwen3.5-397B-A17B Chat (Flash-MoE)\n");
    printf("==================================================\n");
    printf("  Server:  http://localhost:%d\n", port);
    printf("  Session: %s%s\n", session_id, resume_id ? " (resumed)" : "");
    printf("\n  Commands: /quit /exit /clear /sessions\n");
    printf("==================================================\n\n");

    // Health check
    int sock = socket(AF_INET, SOCK_STREAM, 0);
    struct sockaddr_in addr = {0};
    addr.sin_family = AF_INET;
    addr.sin_port = htons(port);
    addr.sin_addr.s_addr = inet_addr("127.0.0.1");
    if (connect(sock, (struct sockaddr *)&addr, sizeof(addr)) < 0) {
        fprintf(stderr, "Server not running on port %d.\n", port);
        fprintf(stderr, "Start it: ./infer --serve %d\n\n", port);
        close(sock);
        return 1;
    }
    close(sock);

    // Resume: load and display previous conversation
    if (resume_id) {
        int turns = session_load(session_id);
        if (turns == 0) {
            printf("No session found with ID: %s\n\n", session_id);
        }
        // Note: server-side KV cache may not match if server restarted.
        // The conversation will continue but model won't "remember" old context
        // unless we re-prefill (TODO: detect server restart and replay).
    }

    printf("Ready to chat.\n\n");

    // Set up linenoise: history, hints
    linenoiseSetMultiLine(1);  // allow multi-line input with arrow keys
    char history_path[1024];
    snprintf(history_path, sizeof(history_path), "%s/.flash-moe/history", getenv("HOME") ?: "/tmp");
    linenoiseHistoryLoad(history_path);
    linenoiseHistorySetMaxLen(500);

    for (;;) {
        char *line = linenoise("> ");
        if (!line) {
            printf("\n");
            break;
        }

        size_t len = strlen(line);
        if (len == 0) { free(line); continue; }

        // Add to history
        linenoiseHistoryAdd(line);
        linenoiseHistorySave(history_path);

        char input_line[MAX_INPUT_LINE];
        strncpy(input_line, line, MAX_INPUT_LINE - 1);
        input_line[MAX_INPUT_LINE - 1] = 0;
        free(line);

        if (strcmp(input_line, "/quit") == 0 || strcmp(input_line, "/exit") == 0) {
            printf("Goodbye.\n");
            break;
        }
        if (strcmp(input_line, "/clear") == 0) {
            generate_session_id(session_id, sizeof(session_id));
            printf("[new session: %s]\n\n", session_id);
            continue;
        }
        if (strcmp(input_line, "/sessions") == 0) {
            session_list();
            continue;
        }

        // Save user turn
        session_save_turn(session_id, "user", input_line);

        sock = send_chat_request(port, input_line, max_tokens, session_id);
        if (sock < 0) continue;

        printf("\n");
        char *response = stream_response(sock, show_thinking);

        // Save assistant turn
        if (response && strlen(response) > 0) {
            session_save_turn(session_id, "assistant", response);
        }

        // ---- Tool call handling ----
        // Detect <tool_call>{"name":"bash","arguments":{"command":"..."}}
        // Execute the command, feed output back as a continuation
        while (response && strstr(response, "<tool_call>")) {
            char *tc_start = strstr(response, "<tool_call>");
            char *tc_end = strstr(tc_start, "</tool_call>");
            if (!tc_start || !tc_end) break;

            // Extract content between tags
            tc_start += 11;  // skip <tool_call>
            char tc_body[4096] = {0};
            int tc_len = (int)(tc_end - tc_start);
            if (tc_len > 4095) tc_len = 4095;
            memcpy(tc_body, tc_start, tc_len);

            // Parse command — handle multiple formats the model might produce:
            // 1. JSON: {"name":"bash","arguments":{"command":"ls -la"}}
            // 2. XML-ish: <function=bash><arg_key>command</arg_key><arg_value>ls -la</arg_value>
            // 3. Simple: just a command string
            char command[4096] = {0};
            int ci = 0;

            char *cmd_key = strstr(tc_body, "\"command\"");
            if (cmd_key) {
                // JSON format: find value after "command":"
                cmd_key = strchr(cmd_key + 9, '"');
                if (cmd_key) {
                    cmd_key++;
                    for (int i = 0; cmd_key[i] && cmd_key[i] != '"' && ci < 4095; i++) {
                        if (cmd_key[i] == '\\' && cmd_key[i+1]) {
                            i++;
                            switch (cmd_key[i]) {
                                case 'n': command[ci++] = '\n'; break;
                                case '"': command[ci++] = '"'; break;
                                case '\\': command[ci++] = '\\'; break;
                                default: command[ci++] = cmd_key[i]; break;
                            }
                        } else {
                            command[ci++] = cmd_key[i];
                        }
                    }
                }
            }
            // Fallback: look for <arg_value>...</arg_value> (model's XML format)
            if (ci == 0) {
                char *av = strstr(tc_body, "<arg_value>");
                if (av) {
                    av += 11;
                    char *av_end = strstr(av, "</arg_value>");
                    if (!av_end) av_end = strstr(av, "<");
                    if (av_end) {
                        int avlen = (int)(av_end - av);
                        if (avlen > 4095) avlen = 4095;
                        memcpy(command, av, avlen);
                        ci = avlen;
                        // Trim whitespace
                        while (ci > 0 && (command[ci-1] == '\n' || command[ci-1] == ' ')) ci--;
                        command[ci] = 0;
                    }
                }
            }
            // Fallback: look for function=bash followed by any command-like text
            if (ci == 0) {
                char *fn = strstr(tc_body, "bash");
                if (fn) {
                    // Take everything after "bash" that looks like a command
                    fn += 4;
                    while (*fn && (*fn == '>' || *fn == '\n' || *fn == ' ' || *fn == '"')) fn++;
                    while (*fn && *fn != '<' && *fn != '"' && ci < 4095) {
                        command[ci++] = *fn++;
                    }
                    while (ci > 0 && (command[ci-1] == '\n' || command[ci-1] == ' ')) ci--;
                    command[ci] = 0;
                }
            }

            if (ci == 0) break;

            // Show the command and ask for confirmation
            printf("\033[33m$ %s\033[0m\n", command);
            printf("\033[2m[execute? y/n] \033[0m");
            fflush(stdout);
            int ch = getchar();
            while (getchar() != '\n');  // consume rest of line
            if (ch != 'y' && ch != 'Y') {
                printf("\033[2m[skipped]\033[0m\n");
                free(response);
                response = NULL;
                break;
            }

            // Execute
            FILE *proc = popen(command, "r");
            char output[65536] = {0};
            int out_len = 0;
            if (proc) {
                while (out_len < 65535) {
                    int ch = fgetc(proc);
                    if (ch == EOF) break;
                    output[out_len++] = (char)ch;
                }
                output[out_len] = 0;
                pclose(proc);
            }

            // Print output
            if (out_len > 0) {
                printf("\033[2m%s\033[0m", output);
                if (output[out_len-1] != '\n') printf("\n");
            }

            // Send tool response back to model as a continuation
            // Format: <tool_response>\n{output}\n</tool_response>
            char *tool_msg = malloc(out_len + 256);
            snprintf(tool_msg, out_len + 256, "<tool_response>\n%s</tool_response>", output);

            free(response);
            sock = send_chat_request(port, tool_msg, max_tokens, session_id);
            free(tool_msg);
            if (sock < 0) { response = NULL; break; }

            printf("\n");
            response = stream_response(sock, show_thinking);

            if (response && strlen(response) > 0) {
                session_save_turn(session_id, "assistant", response);
            }
        }

        free(response);
    }

    return 0;
}
