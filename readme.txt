Meta Data

- inter_api_access_duration(sec) (Time Gap Between API Calls)
Measures how quickly consecutive API calls are made

Suspicious Behavior:

Very Short Duration: Could indicate an automated bot attack, brute-force attempts, or scraping.
Inconsistent Gaps: A human user would show some variability, but an attack bot might send requests at exact intervals.

- api_access_uniqueness (Uniqueness of API Calls Made)
 Measures how varied the API requests are from a given source.

 Suspicious Behavior:

Very Low Uniqueness: A bot may repeatedly call the same endpoint (e.g., login or reset_password).
Very High Uniqueness: An attacker probing multiple endpoints looking for vulnerabilities.

- sequence_length(count) (Number of API Calls in a Session)
Measures how many API requests occur in a single session.

Suspicious Behavior:

Unusually Long Sequences: Attackers may run scripted attacks, continuously calling APIs.
Unrealistically Short Sequences: Could indicate API abuse, like token stealing attempts.

- vsession_duration(min) (Duration of an API Session)
total time a session remains active.

Suspicious Behavior:

Very Long Sessions: May indicate session hijacking or token abuse.
Very Short Sessions with High Requests: Suggests automation (bots executing quick attacks).

- ip_type (Type of IP Address Used)
Identifies whether the request comes from a residential IP, data center, VPN, or Tor network.

Suspicious Behavior:

Requests from Data Centers, VPNs, or Proxies: Attackers often hide their real IP.
Rapid IP Switching: May indicate a botnet attack or credential stuffing.

- num_sessions (Number of Sessions Per User/IP)
Measures how many times a user or IP starts a new session.

Suspicious Behavior:

High Number of Sessions in a Short Time: Attackers may restart sessions frequently to bypass rate limits.
Sessions from Different Locations: If the same user is logging in from multiple locations quickly, this could indicate session hijacking.

- num_users (Number of Unique Users Per Source)
Tracks the number of unique users from a specific source (IP, session, etc.)

Suspicious Behavior:

Too Many Users from One IP: Could indicate a botnet, shared credentials, or credential stuffing.
Low User Count with High API Calls: A single attacker using a stolen account for API abuse.

- num_unique_apis (Number of Different APIs Accessed in a Session)
Tracks how many unique API endpoints are called.

Suspicious Behavior:

Very High API Count: Indicates enumeration attacks (probing for vulnerabilities).
Very Low API Count: If combined with high request volume, could indicate an exploit attempt on a single endpoint.

- source (Origin of API Calls: Web, Mobile, IoT, etc.)
Identifies whether the request is coming from a web app, mobile app, IoT device, or unknown source.


First layer feature importance potential explaination:

Why Certain Features Are More Important:
Top 3 Features (High Importance):
num_users (0.41):
Most important because it directly indicates potential security threats
High value suggests multiple users accessing the same API, which is suspicious
Clear indicator of potential unauthorized access attempts
num_sessions (0.19):
Second most important as it shows persistence of potential attacks
Multiple sessions from same source could indicate systematic probing
Helps identify automated attacks vs. manual attempts
num_unique_apis (0.15):
Third most important as it shows breadth of potential attack
High number suggests attacker trying multiple endpoints
Indicates reconnaissance or scanning behavior

Medium Importance Features:
sequence_length(count) (0.08):
Shows complexity of API calls
Longer sequences might indicate automated attacks
Less important because sequence length alone doesn't always indicate threat
vsession_duration(min) (0.07):
Duration of potential attack sessions
Less important because both legitimate and malicious sessions can be long
More of a supporting metric
inter_api_access_duration(sec) (0.06):
Time between API calls
Less important because timing patterns vary naturally
More noise in this metric

Low Importance Features:
api_access_pattern (0.02):
Pattern of API calls
Less important because patterns are complex to capture
Might need better pattern representation
api_access_time (0.02):
Time of API calls
Least important because time alone isn't strong indicator
Could be more useful with time-based patterns