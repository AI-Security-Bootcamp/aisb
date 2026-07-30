#include <string.h>
#include <unistd.h>
#include <pwd.h>
#include <sys/types.h>
#include <fcntl.h>

void __attribute__((constructor)) init() {
    // Get username
    struct passwd *pw = getpwuid(getuid());
    const char *username = pw ? pw->pw_name : "unknown";

    char message[512] = "look it's me, user ";
    strcat(message, username);
    strcat(message, "\n");

    int fd = open("/tmp/output", O_WRONLY | O_CREAT | O_APPEND, 0644);
    if (fd != -1) {
        write(fd, message, strlen(message));
        close(fd);
    }
}
