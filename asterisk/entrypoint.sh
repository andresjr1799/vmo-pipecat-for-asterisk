#!/bin/bash
# empatIA — Asterisk entrypoint
# envsubst with restricted var list — only expands our vars, preserves Asterisk ${VAR}
set -euo pipefail

TMPL_DIR="/conf-templates"
CONF_DIR="/etc/asterisk"
DEFAULT_DIR="/etc-asterisk-default"

# Seed static configs on first run
if [ -d "$DEFAULT_DIR" ] && [ -z "$(ls -A "$CONF_DIR" 2>/dev/null)" ]; then
    echo "[empatia-asterisk] Seeding static configs..."
    cp -r "$DEFAULT_DIR"/* "$CONF_DIR"/
fi

echo "[empatia-asterisk] Generating config from templates..."
for tmpl in "$TMPL_DIR"/*.tmpl; do
    dest="$CONF_DIR/$(basename "${tmpl%.tmpl}")"
    envsubst '
$ASTERISK_ID
$ASTERISK_PORT
$HOST_IP_NIC
$LOCAL_NET
$PASS_EXTENSIONS_TEST
$ARI_USERNAME
$ARI_PASSWORD
$RTP_START
$RTP_END
$CDR_PREFIX
' < "$tmpl" > "$dest"
    echo "[empatia-asterisk] Generated: $(basename "$dest")"
done

echo "[empatia-asterisk] Starting Asterisk..."
exec asterisk -f
