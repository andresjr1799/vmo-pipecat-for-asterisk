FROM andrius/asterisk:20

USER root

RUN apt-get update && apt-get install -y --no-install-recommends gettext-base \
    && rm -rf /var/lib/apt/lists/*

# Static configs (copied at startup before template expansion)
COPY etc/ /etc-asterisk-default/

# Templates with ${ENV} variables
COPY conf-templates/ /conf-templates/
COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

ENTRYPOINT ["/entrypoint.sh"]
