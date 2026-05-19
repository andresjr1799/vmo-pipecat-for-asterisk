"""ARI event type string constants."""

STASIS_START = "StasisStart"
STASIS_END = "StasisEnd"
CHANNEL_DESTROYED = "ChannelDestroyed"
CHANNEL_VARSET = "ChannelVarset"
CHANNEL_DTMF_RECEIVED = "ChannelDtmfReceived"
CHANNEL_AUDIO_FRAME = "ChannelAudioFrame"
CHANNEL_TALKING_STARTED = "ChannelTalkingStarted"
CHANNEL_TALKING_FINISHED = "ChannelTalkingFinished"

PLAYBACK_FINISHED = "PlaybackFinished"
PLAYBACK_STARTED = "PlaybackStarted"

BRIDGE_CREATED = "BridgeCreated"
BRIDGE_DESTROYED = "BridgeDestroyed"

# Injected by ARIPool before dispatching to handlers
NODE_ID_KEY = "_vmo_node_id"
