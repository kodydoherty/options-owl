#!/usr/bin/env bash
# provision.sh — one-time setup for a fresh Ubuntu 24.04 server.
# Run as root (or with sudo) on the server. Idempotent — safe to re-run.
#
# Usage on the server:
#   curl -fsSL https://your-host/provision.sh | sudo bash
#   # OR after rsyncing the project:
#   sudo bash ~/options-owl/scripts/provision.sh
#
# What this installs:
#   - Docker + compose plugin
#   - Node.js 22 (for Claude Code)
#   - @anthropic-ai/claude-code (CLI)
#   - mosh (resilient mobile SSH)
#   - tmux (persistent sessions)
#   - ufw (firewall — opens 22 for SSH and 60000-61000/udp for mosh)
#   - unattended-upgrades (auto security patches)
#   - A non-root 'owl' user with docker + sudo access
#   - Hardened SSH (disables password login)

set -euo pipefail

OWL_USER="${OWL_USER:-owl}"

echo ">>> Updating apt cache"
apt-get update -y
apt-get upgrade -y

echo ">>> Installing base packages"
DEBIAN_FRONTEND=noninteractive apt-get install -y \
    ca-certificates curl gnupg lsb-release \
    tmux mosh ufw unattended-upgrades \
    git rsync htop jq vim \
    sudo

echo ">>> Installing Docker"
if ! command -v docker >/dev/null 2>&1; then
    install -m 0755 -d /etc/apt/keyrings
    curl -fsSL https://download.docker.com/linux/ubuntu/gpg | gpg --dearmor -o /etc/apt/keyrings/docker.gpg
    chmod a+r /etc/apt/keyrings/docker.gpg
    echo \
      "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] \
      https://download.docker.com/linux/ubuntu $(. /etc/os-release && echo "$VERSION_CODENAME") stable" \
      > /etc/apt/sources.list.d/docker.list
    apt-get update -y
    apt-get install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
    systemctl enable --now docker
fi

echo ">>> Installing Node.js 22 (for Claude Code)"
if ! command -v node >/dev/null 2>&1 || [ "$(node -v | cut -dv -f2 | cut -d. -f1)" -lt 20 ]; then
    curl -fsSL https://deb.nodesource.com/setup_22.x | bash -
    apt-get install -y nodejs
fi

echo ">>> Installing Claude Code CLI"
npm install -g @anthropic-ai/claude-code

echo ">>> Creating user '$OWL_USER'"
if ! id -u "$OWL_USER" >/dev/null 2>&1; then
    useradd -m -s /bin/bash "$OWL_USER"
    usermod -aG sudo,docker "$OWL_USER"
    # Passwordless sudo for the owl user (you'll only ssh in by key anyway)
    echo "$OWL_USER ALL=(ALL) NOPASSWD:ALL" > "/etc/sudoers.d/$OWL_USER"
    chmod 440 "/etc/sudoers.d/$OWL_USER"
fi

# Copy authorized_keys from root if owl doesn't have one yet
if [ -f /root/.ssh/authorized_keys ] && [ ! -f "/home/$OWL_USER/.ssh/authorized_keys" ]; then
    install -d -m 700 -o "$OWL_USER" -g "$OWL_USER" "/home/$OWL_USER/.ssh"
    install -m 600 -o "$OWL_USER" -g "$OWL_USER" \
        /root/.ssh/authorized_keys "/home/$OWL_USER/.ssh/authorized_keys"
fi

echo ">>> Configuring firewall"
ufw --force reset
ufw default deny incoming
ufw default allow outgoing
ufw allow 22/tcp comment 'SSH'
ufw allow 60000:61000/udp comment 'mosh'
ufw --force enable

echo ">>> Hardening SSH (disabling password login)"
sed -i 's/^#*PasswordAuthentication.*/PasswordAuthentication no/' /etc/ssh/sshd_config
sed -i 's/^#*PermitRootLogin.*/PermitRootLogin prohibit-password/' /etc/ssh/sshd_config
systemctl reload ssh || systemctl reload sshd

echo ">>> Enabling unattended security upgrades"
dpkg-reconfigure -f noninteractive unattended-upgrades || true

echo ">>> Creating tmux helper aliases for $OWL_USER"
cat > "/home/$OWL_USER/.bash_aliases" <<'BASHRC'
# OptionsOwl helpers
alias owl='cd ~/options-owl'
alias owl-logs='cd ~/options-owl && docker compose logs -f --tail 100'
alias owl-kody='cd ~/options-owl && docker compose logs -f --tail 100 owlet-kody'
alias owl-status='cd ~/options-owl && docker compose ps'
alias owl-restart='cd ~/options-owl && docker compose restart'
alias owl-up='cd ~/options-owl && docker compose up -d'
alias owl-down='cd ~/options-owl && docker compose down'
alias owl-claude='cd ~/options-owl && tmux new-session -A -s owl "claude"'
alias kill-trades='cd ~/options-owl && bash scripts/kill-trades.sh'
alias start-trades='cd ~/options-owl && bash scripts/start-trades.sh'
BASHRC
chown "$OWL_USER:$OWL_USER" "/home/$OWL_USER/.bash_aliases"

echo ""
echo "================================================================"
echo "Provisioning complete."
echo ""
echo "Next steps:"
echo "  1. From your laptop, rsync the project:"
echo "     bash scripts/deploy.sh $OWL_USER@<server-ip>"
echo ""
echo "  2. SSH in as $OWL_USER:"
echo "     ssh $OWL_USER@<server-ip>"
echo ""
echo "  3. First-time Claude Code login (browser flow):"
echo "     claude    # follow the URL it prints"
echo ""
echo "  4. Start the bots:"
echo "     owl-up"
echo ""
echo "Emergency commands (from any SSH session, including iPhone):"
echo "  kill-trades       # flip kill switch on, restart kody"
echo "  start-trades      # flip kill switch off, restart kody"
echo "  owl-kody          # tail kody logs live"
echo "  owl-claude        # attach to persistent Claude Code tmux session"
echo "================================================================"
