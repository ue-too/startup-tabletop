# STARTUP SIMULATOR: MAIN RULEBOOK (Version 6.0)

## 1. OBJECTIVE

You are a startup founder in a volatile economy. Your goal is to build the highest **Company Valuation** before the "Market Crash" card is drawn. You achieve this by launching products, dominating the market through licensing, and building a portfolio of strategic investments.

---

## 2. COMPONENTS

### Decks
* **Talent Deck:** Senior & Specialist Employees (Face Down).
* **Product Decks (Split):**
    * **Seed Deck:** Tier 1 Products only.
    * **Growth Deck:** Tier 2 & Tier 3 Products + "Market Crash" Card.
* **Event Deck:** Global modifiers (Face Down).
* **Strategy Deck:** Action Cards for Training, PR, and Sabotage (Face Down).
* **Milestones:** Public Objectives (Face Up).

### Tokens
* **Currency:** Capital ($).
* **Work Cubes:** **Blue** (Code {}) and **Red** (Components [Chk]).
* **Status Tokens:** Black (Bug), Green (Hype), Red (Scandal).
* **Upgrade Tokens:** White (XP), Gold (Rank Badge), Blue/Red Discs (Skill).
* **Corporate:** Equity Tokens (3 per player), Market Share Tokens.

---

## 3. KEY CONCEPTS

### The Zones & Upkeep
1.  **The Bench (Hand):** Employees hired but not assigned.
    * **Upkeep:** **$0** (Free to hold).
    * **Hand Limit:** Max **5 Cards** at end of turn.
2.  **Development Zone:** Products under construction. Staff are assigned *directly* to these cards.
    * **Capacity Limit:** Maximum **3 Active Projects**. You cannot Greenlight a 4th project until you clear a slot (via Launch or Pivot).
    * **Upkeep:** Must pay Salaries.
    * **Value:** Products here have a liquidation value of **$0**.
3.  **Ops Zone (Maintenance):** Staff pool running active products.
    * **Upkeep:** Must pay Salaries.
    * **Security:** **SAFE ZONE.** Staff in the Ops Zone **cannot be Poached**.

### Talent Growth (XP System) see section 11
* **Juniors Only:** Only cards marked "Junior" can gain XP. Seniors are static.
* **Pending XP:** XP is earned on the *first turn* a Junior contributes. It is placed in a **"Pending"** state (limit 1 per kind (software, hardware, and qa) per Junior) until the project Launches.
* **Max Potential:** A Junior can hold max **4 XP Tokens**.
    * *Base Output:* 1.
    * *Max Output:* 5 (1 Base + 4 XP).
* **Salary Threshold:**
    * 0-1 XP: **$0 Salary**.
    * 2+ XP: **$1 Salary**.

### Tech Tiers & Leads
* **Tier 1:** Can be built by any team.
* **Tier 2 & 3:** Must have a **Matching Lead** (Tier 2+ Talent) assigned.
    * **Matching Rule:**
        * **Software Product `{}`:** Needs Software Lead.
        * **Hardware Product `[Chk]`:** Needs Hardware Lead.
        * **Hybrid Product:** Needs *either* to progress, but *both* to Launch.
* **Stalled:** If the Lead does not match the Product Type, the team produces **0 Output**.
* **Cross-Functional Leads:** Staff with **Both Skills** (Firmware, Full Stack, or Dual-Skilled Rank Badge holders) count as Matching for *any* project.

---

## 4. SETUP

1.  **The Markets:**
    * **University:** Place Junior Stacks face up.
    * **Agency Row:** Deal 4 Talent Cards face up.
    * **Product Market:** Deal 2 Seed Cards (Left) and 2 Growth Cards (Right).
    * **Open Job Market:** Designate space for a queue of 5 cards.
    * **Open Idea Pool:** Designate space for a queue of 5 cards.
2.  **Global:** Reveal 3 Milestones. Shuffle **"Market Crash"** into the bottom 20% of the Growth Deck.
3.  **Player Setup:**
    * **Capital:** **$7** (Seed Funding).
    * **Equity:** 3 Tokens of your color.
    * **Starting Hand:** 1 Jr Software, 1 Jr Hardware, 1 Jr QA.
    * **Seed Round:** Draft 1 "Concept" Product to Dev Zone.

---

## 5. TURN STRUCTURE

### Phase A: Event
Reveal 1 Event Card. Apply effects globally.

### Phase B: Income (The Ops Check)
1.  **Supply vs. Demand:** Sum total Bandwidth in Ops Zone vs. Maintenance Costs of Products.
    * **Bandwidth Calculation:** Ops Bandwidth is calculated the same as Dev Output: Sum all staff Output values (Base + XP + Bonuses) in their declared modes. Staff in Ops Zone provide Bandwidth matching their Output type (Blue `{}` or Red `[Chk]`).
    * If Supply < Demand, choose products to go **Offline** ($0 Rev).
    * **Note:** If Ops Bandwidth < Maintenance Costs (including acquired products), choose which products go Offline using standard rules.
2.  **Operational Revenue:** Sum Revenue of Active Products.
    * *Bonuses:* Add Sales/Integration bonuses.
    * *Decay:* Subtract **$1 per Bug Token** on Active Products (Min $0).
3.  **Trigger Dividends**: The Bank pays your investors based on your Operational Revenue tier (See Section 9).
4.  **Collect Financial Income:** Collect Dividends from Equity you hold.
5.  **Salaries:** Pay for **all** Board Staff (Dev + Ops). Discard unpaid staff to Open Job Market.

### Phase C: Actions (3 AP)
Spend 3 Action Points. You may repeat actions. (See Section 6).

### Phase D: The Engine & Audit (Detailed)
1.  **Generate & Train:** Active Dev Teams generate Cubes.
    * **Transient Zone:** Place newly generated cubes **next to** the card (not on the tracks yet).
    * **Pending XP:** If a Junior generates a cube and has **NO Pending Token**, place a matching XP Token (Blue/Red) on them as **Pending**.
2.  **Clean & Train:** QA Staff in Ops remove Bugs.
    * **Pending XP:** If a Junior acts as QA and has **NO Pending Token**, place a **Green XP Token** on them as **Pending**.
3.  **THE AUDIT WINDOW (Auction):**
    * Opponents may bid (Start $3) to flip/audit a **Face-Down** card.
    * *Check:* Legality of **Transient Cubes** and **Dependency Tags**.
    * *Resolution:* See Section 7. (and for more details on audits)
4.  **Commit:** Move Legal cubes from Transient Zone -> Product Tracks.
5.  **Complete:** Mark full products as "Feature Complete."
6.  **Refill:** Markets (Agency & Product Row).
7.  **Cleanup:**
    * Enforce Hand Limit (5) -> Discard to Open Job Market.
    * Enforce Strategy Hand Limit (3) -> Discard.
    * Enforce Product Backlog Limit (3) -> Discard to Open Idea Pool.
    * **Market Caps (FIFO):** If Job/Idea pools have > 5 cards, remove Oldest.

## Sidenotes: PLAYER AIDS & UX DESIGN (v2.0)

To reduce Analysis Paralysis, print these updated visual aids for your players.

### 1. THE "PIPELINE" PLAYER MAT (Layout Guide)
*This layout groups actions by department. It separates **Paid Actions (1 AP)** from **Free Actions (0 AP)**.*

**VISUAL LAYOUT:**

| **DEPARTMENT 1: HR (Talent)** | **DEPARTMENT 2: R&D (Product)** | **DEPARTMENT 3: ADMIN (Finance)** |
| :--- | :--- | :--- |
| **1. GET RESOURCES** | **1. GET IDEAS** | **1. GET CAPITAL / CARDS** |
| [ ] **Recruit** (Hire to Bench) | [ ] **Ideation** (Draft to Backlog) | [ ] **Invest** (Buy Equity) |
| [ ] **Recall** (Ops -> Bench) | | [ ] **Divest** (Sell Assets) |
| [ ] **Layoff / Source** (Cycle Deck) | | [ ] **Brainstorm** (Draw Strategy Cards) |
| | | |
| **2. DEPLOY RESOURCES** | **2. MANAGE WORK** | **2. AGGRESSION** |
| [ ] **Assign** (Bench -> Board) | [ ] **Greenlight** (Start Dev) | [ ] **Hostile Offer** (Poach) |
| [ ] **Reassign** (Move on Board) | [ ] **Pivot** (Scrap Project) | [ ] **Acquisition** (Buy Maint Product) |
| | [ ] **Launch** (Dev -> Maint) | |

---

#### **FREE ACTIONS (0 AP)**
*These can be done at any time during your turn.*

* **Play Strategy Card:**
    * **Training:** Apply Skill/Rank/XP to Talent.
    * **PR:** Add Hype/Scandal.
    * **Sabotage:** Lawsuits/Attacks.
* **Integrate:** Stack Host + Client products.
* **Voluntary Disclosure:** Flip Stealth Card (Pay Late License).

### 2. THE S.O.P. CARD (Standard Operating Procedures)
*Print this on a small card and give it to new players. It provides "Default Turns" so they don't freeze up.*

#### STUCK? TRY ONE OF THESE TURNS:

**OPTION A: THE BUILDER (Early Game)**
*Focus: Fishing for Skills.*
1.  **Recruit:** Hire a Junior from University ($2).
2.  **Brainstorm:** Dig for a "Bootcamp" or "Certification" card.
3.  **Assign:** Put the Junior on a project (even if untrained).
    * *Note:* Use the Training card immediately (0 AP) if you drew one!

**OPTION B: THE FOUNDER (Mid Game)**
*Focus: Filling the pipeline.*
1.  **Ideation:** Draft a Tier 2 "Enabler" card.
2.  **Greenlight:** Start building it (Check your Tags!).
3.  **Brainstorm:** Look for "Design Sprint" or "PR" cards.

**OPTION C: THE SHARK (Aggressive)**
*Focus: Disruption.*
1.  **Invest:** Buy Equity in the current Leader ($10/$15).
2.  **Hostile Offer:** Poach their Lead Developer (2x Cost).
3.  **Brainstorm:** Dig for a "Hit Piece" to scandalize them later.

**OPTION D: THE PIVOT (Recovery)**
*Focus: Fixing a broken board.*
1.  **Recall:** Pull your Ops Team back to the Bench.
2.  **Pivot:** Scrap a failed Development project.
3.  **Layoff / Source:** Discard the bad staff to find Seniors in the deck.

---

## 6. ACTIONS MENU (1 AP Each)

### Category: Talent
* **Recruit:** Buy 1 Talent to **Bench** or **Product** Zone.
    * *University:* $2.
    * *Agency:* Printed Cost.
    * *Open Market:* $1.
* **Assign:** Move cards from Bench -> Board (Dev or Ops). **any number of talents** can be assigned at once (for 1 AP). Assign employees who generate the cube types your product needs otherwise all the progress work cubes go to waste. Choose wisely for who you want to develop your product.
* **Recall (Mass Retreat):** Move **any number** of cards from the **Ops Zone** $\rightarrow$ **Bench**.
    * *Use Case:* Use this to pull a Maintenance team back to your hand so you can re-deploy them to a new Development project on your next action.
    * *Restriction:* You cannot Recall from an Active Development project (You must **Pivot/Scrap** to get them back).
* **Reassign:** Move 1 Talent between Board Teams.
    * **Standard Penalty:** Place **Onboarding Token** (0 Output this turn). This applies to all personnel, not just the developers. The onboarding token cancels all the effects of the talent for that turn.
    * **Agile Exception:** If moving **TO** a team with a **Senior PM**, do **NOT** place an Onboarding Token.
* **Layoff / Source:** Discard X cards to Open Market -> Reveal X cards from Deck -> Hire or Pass.

### Category: Product
* **Ideation:** Draw 1 Product to **Backlog** (Hand).
    * *Source:* Market Row (1 AP), Decks (1 AP), or Open Idea Pool (Cost **2 AP**).
    * **The Overflow Flush:** If you already have 3 cards, you may still Draft a 4th. This forces you to discard one card to the Open Idea Pool during the Cleanup Phase (effectively costing 1 AP to flush a bad card).
    * *Restriction:* You cannot voluntarily discard a card to the Open Idea Pool. You must either flush it or greenlight it.
* **Greenlight (0 AP):** Move Backlog -> Development.
    * **Condition:** You must have an open slot in the Development Zone (Max 3).
    * *Stealth:* Face-Down if you have **Secured Access** (Tag Owned or Equity).
    * *Dependency:* Pay **$3 License** if missing Tag (must be Face-Up).
* **Launch:** Dev -> Maintenance.
    * **Req:** Full Cubes + 0 Bugs.
    * **Movement:** Move the Product Card **AND all assigned staff** from Development Zone to Ops Zone.
    * **Lead Check (Tier 2/3):**
        * **Pure Project:** Must have 1 Matching Lead.
        * **Hybrid Project:** Must have **Full Leadership Coverage** (1 Soft Lead AND 1 Hard Lead) **OR** (1 Cross-Functional Lead).
    * **XP Graduation:** All **Pending XP** tokens on the team become **Permanent**.
    * **Stealth Bonus:** Tier 2 (+5 VP), Tier 3 (+10 VP).
    * **Note:** Launching is optional. You may keep a completed product in Development as long as desired, but it generates no revenue and you must continue paying staff salaries.
* **Pivot:** Scrap Project.
    * *XP Loss:* Discard all **Pending XP** tokens. Return staff to Bench. (Benched staff are subject to the onboarding token penalty for that turn. Also if not reassigned to a new project, they are subject to the cleanup phase, only 5 cards are allowed in the hand at the end of the turn.)
* **Acquisition:** Buy Opponent's Maint Product. (See Section 7).
* **Voluntary Disclosure (0 AP):** You may flip a Stealth card Face-Up at any time.
    * *Effect:* You **Lose** the Stealth/Keynote Bonus potential.
    * *Legality Check:*
        * **Secured Access:** If you Own the Tag or hold Equity, the reveal is **Free**.
        * **Unsecured (Late License):** If you need a Tag from a non-partner, you must pay a **$4 Late License** ($3 to Tag Owner + $1 to Bank).
    * *Use Case:* Do this to avoid an Audit Risk ($6 Settlement) or to secure a License before a partner sells their tech. This can also be used to secure and lock in the legal status of the product.

### Category: Management
* **Brainstorm:**
    1.  **Flush (Optional):** Discard any number of Strategy Cards from your hand.
    2.  **Draw:** Draw **2 Cards** from the Strategy Deck.
    3.  **Keep:** Choose **1 Card** to keep. Discard the other.
    * *PM Bonus:* If you have a PM, Draw 3 / Keep 1.
* **Invest (Primary Market):** Pay fixed price for Equity.
* **Secondary Trade:** Pay fixed price to buy a Founder's Equity from *another Investor*.
    * *Condition:* Seller must **Consent**. Founder cannot block this.
* **Play Strategy Card (0 AP):** Play "Training", "PR", "Sabotage" cards.
    * *Training:* Use cards like "Bootcamp" ($3) to give Juniors new skills (Tokens).
* **Integrate (0 AP):** Connect Host/Client products in Maintenance.
* **Buyback:** Pay fixed price to repurchase your own Equity from an Investor.
    * *Condition:* Investor must **Consent**.
* **Divest (Sell Asset):** Sell an asset to an opponent to raise cash.
    * **Target:** An Equity Token you hold OR a Maintenance Product you own.
    * **Condition:** Buyer must **Consent** and have funds/bandwidth.
    * **Price:** Fixed Market Price (Share Price or M&A Price).

### SUMMARY: WHO PAYS THE AP?

| Scenario | Whose Turn? | Action Name | Who Pays AP? | Why? |
| :--- | :--- | :--- | :--- | :--- |
| **Buyer wants the Asset** | Buyer's | **Acquire / Trade** | **Buyer (1 AP)** | The Buyer wants the VP, Perks, or Revenue immediately to advance their strategy. |
| **Seller wants Cash** | Seller's | **Divest** | **Seller (1 AP)** | The Seller needs immediate Cash to fund other actions (e.g., Hiring, Settlements). |
| **Bankruptcy** | N/A | **Liquidation** | **None (Forced)** | This is a forced game mechanic triggered by debt, not a voluntary action. |

---

## 7. CONFLICT & ECONOMY

### THE AUDIT (CORPORATE ESPIONAGE)

The Audit is a high-stakes mechanism to catch players bluffing or cheating on their "Stealth" (Face-Down) products. It occurs in **Phase D: Step 3**, immediately after players place cubes into the **Transient Zone**.

#### A. THE BIDDING WAR
Any opponent may initiate an Audit on the Active Player's **Face-Down** cards.

1.  **Declaration:** An opponent points to a specific Face-Down card and says "I want to Audit."
2.  **The Auction:** All opponents (excluding the Active Player) may bid cash for the right to perform the audit.
    * **Minimum Bid:** **$3**.
    * **Winner:** The highest bidder becomes the **Whistleblower**.
3.  **The Reveal:** The Whistleblower flips the target card Face-Up.

#### B. THE LEGALITY CHECKLIST
The table immediately checks the validity of the card against the **Transient Cubes** (cubes added *this turn*) and the **Board State**.

**The project is ILLEGAL (Fraudulent) if ANY of the following are true:**

1.  **Ghost Progress (Tier 2/3 Only):**
    * Project has Transient Cubes but **No Matching Lead** (Tier 2+).
      * Hardware Project requires Tier 2+ with hardware skill.
      * Software Project requires Tier 2+ with software skill.
      * Cross-Functional Tech Lead can be used to satisfy both requirements. (At the same time)
    * *Rule:* You cannot generate work on a complex project without a Lead.
2.  **Missing Dependency:**
    * The product requires a Tag (e.g., `[AI]`), AND...
    * The Owner does **NOT** have the Tag in their Maintenance Zone, AND...
    * The Owner does **NOT** hold an Equity Token of a player who has the Tag.

**Summary: Who Can Lead What? (Tier 2/3 Rules)**

| Talent Type | Can Lead Software? | Can Lead Hardware? | Can Lead Hybrid? |
| :--- | :--- | :--- | :--- |
| **Senior Backend** (Software) | **YES** | NO (Stalled) | **Partial** (Can Build, Can't Launch) |
| **Senior Hardware** (Hardware) | NO (Stalled) | **YES** | **Partial** (Can Build, Can't Launch) |
| **Firmware / Full Stack** (Cross-Functional) | **YES** | **YES** | **FULL** (Can Build & Launch) |
| **Junior + Rank Badge** | Only if has Software Skill | Only if has Hardware Skill | Only if has **Both** Skills |

> **Note on Wrong Colors:** Placing "Wasteful" resources (e.g., Red Cubes on a Blue-only product) is **LEGAL**. It is poor strategy, but not fraud. The cubes simply fail to advance the project.

> **Note:** If the project has "Legacy Cubes" (cubes from previous turns) that are wrong, but the *current* turn's Transient Cubes are correct, the Audit focuses only on the **current turn's legality** regarding Ghost Progress. However, Missing Tags make the whole project Illegal regardless of turn.

#### C. THE VERDICT & RESOLUTION

Follow this decision tree to resolve the money and the card status.

##### SCENARIO 1: THE PROJECT IS LEGAL (Clean)
The Whistleblower was wrong. The startup is legitimate.

1.  **The Penalty:** The Whistleblower pays their **Full Bid Amount** directly to the **Owner**.
2.  **The Project:** The Card remains **Face-Up**.
    * *Consequence:* Since the card is now Face-Up, the Owner loses the potential **Stealth Launch Bonus** (+5/+10 VP) for the future, but the project continues.
3.  **The Cubes:** Transient Cubes are successfully committed to the tracks.

##### SCENARIO 2: THE PROJECT IS ILLEGAL (Fraud)
The Whistleblower caught the Owner lying.

1.  **Bid Return:** The Whistleblower takes their **Full Bid Amount** back from the table (Cost = $0).
2.  **Availability Check:** The table checks if the required Tag exists *anywhere* in a Maintenance Zone (Global Market).
    * **If Tag DOES NOT Exist:** The Owner has no choice. Proceed to **FOLD**.
    * **If Tag DOES Exist:** The Owner chooses to **FOLD** or **SETTLE**.

**Option A: FOLD (The Project Collapses)**
* **The Project:** Scrapped immediately. Discard the Card and all Cubes (Legacy and Transient). Return Staff to Bench.
* **The Fine:** The Owner pays **$5 Total**:
    * **$4** goes to the **Whistleblower**.
    * **$1** goes to the **Bank**.

**Option B: SETTLE (Emergency Legalization)**
* **The Project:** Survives. It becomes **Legal** immediately. Transient Cubes are committed.
* **The Settlement Cost:** The Owner pays **$6 Total**:
    * **$3** goes to the **Tag Owner** (or Bank if settling a Ghost Progress error).
    * **$3** goes to the **Bank** (Legal Fees).
    * *(Note: If the Owner holds Equity in the Tag Owner, they pay $0 for the Tag, but must still pay the $3 Bank Fee).*
* **The Bounty:** The **Bank** pays the Whistleblower a **$4 Reward** for exposing the issue.

#### D. SUMMARY TABLE: WHO PAYS WHOM?

| Outcome | Whistleblower Pays | Owner Pays | Whistleblower Gets | Project Status |
| :--- | :--- | :--- | :--- | :--- |
| **LEGAL** | Bid $\rightarrow$ Owner | $0 | Nothing | Continues (Face Up) |
| **ILLEGAL (Fold)** | $0 (Bid Returned) | \$4 $\rightarrow$ WB<br>\$1 $\rightarrow$ Bank | **$4** (From Owner) | **Scrapped** |
| **ILLEGAL (Settle)**| $0 (Bid Returned) | \$3 $\rightarrow$ Tag Owner<br>\$3 $\rightarrow$ Bank | **$4** (From Bank) | Continues (Face Up) |

#### E. SPECIAL CASE: THE BROKEN CHAIN (Liquidation Event or Transfer Event of a Product)

If a Tag Source (e.g., a Partner's Platform) is **Liquidated/Sold** to the Bank/Player, that Tag ceases to exist on the board (sold to the bank) or you lose access to it (sold to a player). Any Stealth Products relying on that specific Tag become instantly **Vulnerable**. 

**The "Come Clean" Protocol (Fire Sale):**

Upon the moment of Liquidation, the Owner of the vulnerable Stealth Product may choose to **Voluntarily Reveal** (Flip Face-Up) their card immediately.

* **If Revealed:** The product gains **Legacy Status** for the *specific tag that was lost*. It is considered permanently satisfied for that one requirement.
* **If Kept Hidden:** The Owner effectively claims they have a different valid source. If Audited later, they must prove valid access.

**Crucial Restriction:**
"Coming Clean" **ONLY** creates an exception for the Tag that was just sold.
* If the product requires `[AI]` + `[Cloud]`, and the partner sold `[AI]`:
    * Revealing grants valid status for `[AI]`.
    * **HOWEVER**, if the Owner *also* lacks `[Cloud]`, the project remains **ILLEGAL** upon reveal.
* *Result:* An illegal project revealed during a "Come Clean" attempt is treated as a **Failed Audit** (Owner must Fold or Settle immediately).

**The "Come Clean" Protocol (Market Sale):**

The timing and rules are the same as the fire sale. But the cost is $3 (to the new owner) instead of $0.
* *Scenario:* Partner sold the product to an opponent you do not hold Equity in.
* **Resolution:** Flip Face-Up. You must immediately **Pay the $3 License Fee** to Player C.
    * *Constraint:* If you cannot afford the $3, you **cannot** Come Clean. You must remain Face-Down (and risk an Audit for Fraud).

### Liquidation (M&A)
**M&A Price (Fixed):**
* **Base:** Tier 1 ($6) | Tier 2 ($12) | Tier 3 ($20).
* **Modifiers:** Hype (+$5), Attached Specialist (+$5), Scandal (-$5), Bug (-$1).

**What is INCLUDED (Transfers to Buyer):**
* **The Card:** Moves to Buyer's Maintenance Zone.
* **Tokens on Card:** Hype, Scandal, and Bug tokens move with the card.
* **Attached Staff:** Specialists attached to the card move to the Buyer.

**What is EXCLUDED (Stays with Seller):**
* **Market Share Tokens:** These represent past earnings. The Seller **KEEPS** all Market Share tokens generated by this product. They do not affect the price.
* **Legacy Status:** If the product had "Legacy Status" (from a Broken Chain event), that status is reset. The Buyer must satisfy the dependency anew (or have their own tags) if they ever need to re-verify legality.

**Rules:**
**2. Attempt Market Sale (Preferred):**
* Offer the product to opponents in Turn Order.
* **Price:** Buyer pays **100% of M&A Price**.
* **Condition:** Buyer must have Bandwidth to run the product immediately.
* *Effect:* Buyer receives Card + Attached Staff. Seller gets full cash.

**3. Fire Sale (Last Resort):**
* If no player buys the product (or none have bandwidth), sell to the **Bank**.
* **Price:** Bank pays **50% of M&A Price** (Rounded down).
* *Effect:* Product is Scrapped (moved to Open Idea Pool). Staff return to Seller's Bench.

### Investment Perks
If Player A holds Equity in Player B:
1.  **Dividends:** A collects $1-$3 based on B's Revenue.
2.  **Immunity:** A cannot Poach B.
3.  **Reciprocal Licensing:** A and B use each other's Tags for **Free**.
4.  **Consultant:** Equity Token adds +1 Output to specific card.
5.  **Veto:** Burn Token to Cancel Action.

### THE HEADHUNTER (Card-Based Poaching)
Poaching is no longer a standard action. To steal an employee, you must play a **"Headhunter"** Strategy Card from your hand.

**1. The Play:**
* **Action:** Reveal "Headhunter" card (0 AP).
* **Target:** Choose an opponent's Board Staff (Dev or Ops).

**2. The Cost:**
* Pay the **Bank** the Poaching Fee.
* **Standard Cost:** `2x (Base Cost + Token Value)`
* *Note:* Some advanced Headhunter cards might offer a discount (e.g. 1.5x), but the standard rate is 2x.

**3. Resolution:**
* **Check Defenses:** Fails if HR Manager, Immunity, or Vested.
* **Transfer:** Move card to your Board.
* **Orientation:** Place **Sideways (Tapped)**. (Produces 0 Output this turn; Immune to poaching until straightened).

---

## 8. ADVANCED MECHANICS

### Ecosystem Integration (refer to section 10 for details)

### The Stock Market
**Buy Price:**
Based on Company's Highest Active Tier:
* **Tier 1:** $5
* **Tier 2:** $10
* **Tier 3:** $15

**Dividend Payout:**
Based on Founder's Operational Revenue:
* **$0 Rev:** $0 Dividend.
* **$1 - $10:** $1 Dividend.
* **$11 - $20:** $2 Dividend.
* **$21+:** $3 Dividend.

---
## 9. DIVIDENDS & REVENUE CALCULATION

Dividends are the payouts investors receive based on the performance of the companies they invested in.

### A. THE "OPERATIONAL REVENUE" RULE
To determine how much a company pays out, you must calculate its **Operational Revenue**. This is a specific number derived *only* from product performance.

**The Formula:**
> **Operational Revenue = (Sum of Active Product Revenue + Staff Bonuses) - (Bug Decay)**

**What is INCLUDED:**
* **Base Revenue:** The printed `Rev` value on active Maintenance cards.
* **Bonuses:** +$ from Sales Reps, Growth Hackers, or Ecosystem Integration.
* **Decay:** The subtraction of $1 per Bug Token.

**What is EXCLUDED (Crucial):**
* **Financial Income:** Dividends you collect from *other* players do **NOT** count as Revenue. You cannot pay dividends using money earned from dividends (no Ponzi schemes).
* **Cash on Hand:** Money saved from previous turns does not count.
* **Development Cards:** Products in development generate $0.

### B. THE DIVIDEND PAYOUT TABLE (Yield)
Once a player's Operational Revenue is calculated, look up the **Yield Tier** below.

* **Who Pays:** The **BANK** pays the Investor. (This represents the stock market distributing value, not money taken directly from the Founder's pocket).*
* **Who Receives:** Any opponent holding that Founder's **Equity Token**.

| Founder's Operational Revenue | Dividend Paid (Per Token) |
| :--- | :--- |
| **$0 (Pre-Revenue)** | **$0** |
| **$1 - $10** | **$1** |
| **$11 - $20** | **$2** |
| **$21+** | **$3** (Max Cap) |

*( *Design Note: Having the Bank pay ensures the Founder is not "punished" for being successful. If the Founder had to pay $3 to two investors out of their own $21 revenue, it would discourage growth. This method keeps the game positive-sum.)*

### C. EXAMPLE SCENARIO

**Player A (Founder)** has:
1.  **Product:** Search Engine (Base Rev $5).
2.  **Staff:** Sales Rep attached (+$2 Bonus).
3.  **Status:** 2 Bug Tokens on the card (-$2 Decay).
4.  **Portfolio:** Player A also holds equity in Player B, earning them $2 Financial Income.

**The Calculation:**
1.  **Base:** $5
2.  **Bonus:** +$2
3.  **Decay:** -$2
4.  **Operational Revenue:** $5.
    * *(Note: The $2 from Player B is ignored).*

**The Payout:**
* $5 falls into the **$1-$10 Tier**.
* **Result:** Anyone holding Player A's Equity Token receives **$1 from the Bank**.

---

## 10. ECOSYSTEM INTEGRATION (The Walled Garden)

Players can physically link compatible products in their **Maintenance Zone** to create a "Vertical Monopoly" (e.g., loading your own App onto your own Device). This is known as a **Product Stack**.

### A. THE MECHANIC
* **Action:** **Integrate (0 AP)**. This is a free action taken during your turn.
* **Execution:** Choose a **Host Card** and a **Client Card** in your Maintenance Zone. Slide the Client Card partially underneath the bottom edge of the Host Card.
* **Limit:** A Host can support only **1 Client** (forming a pair). A Client cannot be a Host to another card (no daisy-chaining).

### B. VALID PAIRINGS (Compatibility)
To **Integrate (Stack)** two products, they must have a valid Host/Client relationship.

**1. HARDWARE HOSTS** (`[Device]`, `[IoT]`, `[Robotics]`)
* **Can Host:** `[App]`, `[Data]`, `[Media]`, `[AI]`.
* *Logic:* "Software running on Hardware."

**2. DIGITAL HOSTS** (`[Platform]`, `[Cloud]`, `[Network]`)
* **Can Host:** `[Service]`, `[FinTech]`, `[Crypto]`, `[Media]`.
* *Logic:* "Services running on Infrastructure."

**3. UNICORN HOSTS** (Tier 3 Special)
* **Metaverse:** Hosts `[Social]`, `[Media]`, `[Commerce]`.
* **Mars Colony:** Hosts `[Bio]`, `[Energy]`, `[IoT]`.

**RESTRICTIONS:**
* A Host cannot hold more than **1 Client**.
* A Client cannot be a Host (No daisy-chaining).
* You cannot stack two cards of the **same Tag** (e.g., Cloud on Cloud).

### C. THE ECOSYSTEM BONUSES
As long as the two cards remain stacked, they generate special bonuses that override standard rules.

**1. Revenue Boost (The Walled Garden):**
The **Client (Child)** generates **+$2 Revenue** during the Income Phase.
* *Theme:* "Pre-installed apps have zero user acquisition cost."

**2. Stickiness (Churn Defense):**
The **Client (Child)** ignores the **Revenue Decay** rule.
* *Effect:* Bug Tokens on the Client do **NOT** reduce its revenue. (Bugs on the Host still apply normal decay).
* *Theme:* "Users can't switch apps because they are locked into our ecosystem."

**3. Valuation Multiplier (The IPO Story):**
At the end of the game, the **Host (Parent)** gains **+5 VP** to its Valuation.

### D. BREAKING THE STACK
If either card is Sold (Acquisition/Liquidation) or Scrapped:
1.  **The Stack Breaks:** The remaining card returns to being a standalone product.
2.  **Loss of Bonus:** All Revenue, Stickiness, and Valuation bonuses cease immediately.

### E. ACTIVE STATUS REQUIREMENT
Integration bonuses are **conditional**. They only apply during the Income/Scoring phase if **BOTH** the Host and the Client are **Active** (within Bandwidth Limits).

1.  **Broken Link:** If *either* card is **Offline** (due to lack of Ops Bandwidth), the connection is severed for that turn.
    * **Revenue Bonus:** Lost.
    * **Stickiness (Bug Immunity):** Lost. (Bugs apply decay to the Client immediately).
    * **Valuation:** If the link is broken during the Final Scoring step, the +5 VP is lost.

2.  **Strategy Note:** Players should group Integrated Strips together at the **bottom** of their Server Rack to ensure the connection stays live.

### F. END GAME SCORING & FAILURE STATES

For an Ecosystem Stack to generate the **+5 VP Bonus** during the Final Valuation, the connection must be **Live**.

**The Rule:**
If **EITHER** the Host Card or the Client Card is **Inactive (Offline)** due to lack of Bandwidth at the moment of game end, the Ecosystem is broken.

**Scoring Scenarios:**

1.  **Scenario A: The Client Goes Offline**
    * *Cause:* The Client strip was pushed above the Capacity Slider in the Server Rack.
    * **Client Score:** **0 VP** (Standard rule: Inactive products score nothing).
    * **Host Score:** **Base VP Only**. (The +5 Bonus is lost because the ecosystem has no active user base).

2.  **Scenario B: The Host Goes Offline**
    * *Cause:* The Host strip was pushed above the Capacity Slider.
    * **Host Score:** **0 VP** (Standard rule: Inactive products score nothing. The Bonus is irrelevant because the base is 0).
    * **Client Score:** **Base VP Only**. (It survives as a standalone app, but loses the ecosystem "Stickiness" and Revenue protection).

3.  **Scenario C: The "Perfect Stack"**
    * *Condition:* Both strips are completely below the Capacity Slider.
    * **Result:** Client scores Base VP + Host scores Base VP + **Host gains +5 VP Bonus**.

---

## 11. TALENT GROWTH & SKILLS

In *Startup Simulator*, employees are not static assets. Juniors evolve into superstars based on **what they actually do**.

### A. THE JUNIOR LIFECYCLE (Organic Growth)
Only cards marked **"Junior"** have the ability to learn "on the job."

1.  **The Base Skills:**
    * Every Junior starts with **1 Native Skill** (Software or Hardware) at **Output 1**.
    * Acquiring a **New Skill Token** (via Training) grants a **Base Output of 1** in that new skill immediately.
2.  **The Total Cap:**
    * A Junior can hold a maximum of **4 Permanent XP Tokens** total.
    * *Max Potential:* **3 Base Skills** + **4 XP Tokens** = **7 Total Skill Points**.
3.  **The Price of Success (Salary Bump):**
    * **Rookie (0-1 XP):** Salary is **$0**.
    * **Pro (2+ XP):** Salary becomes **$1**.

### B. THE CONTRIBUTION RULE (Accumulated Learning)
Since development takes multiple turns, a Junior might perform different tasks (coding vs. bug fixing). We use **"Pending Tokens"** to track *everything* they did, and then filter it at Launch.

**1. The "Pending" State (Accumulation)**
* **Trigger:** When a Junior generates a Cube or Removes a Bug during the **Engine Phase**.
* **The Check:** Does this Junior already have a **Pending XP Token** of that *specific color*?
    * **NO:** Place a matching token (Blue, Red, or Green) on the card as **"Pending."**
    * **YES:** Do nothing. (You cannot gain two Blue Pending tokens).
* **The Limit:** A Junior can hold up to **3 Pending Tokens** at once (1 Blue, 1 Red, 1 Green), representing a diverse contribution to the project.

**2. Graduation (The Launch)**
* **The Decision:** Upon a successful Launch, look at the Pending Tokens on the Junior.
* **Selection:** The Player must choose **EXACTLY ONE** Pending Token to make **Permanent**.
    * Move the chosen token to the specific Skill Slot (Blue/Red/QA). The bonus +1 Output is now active.
* **Cleanup:** All *other* Pending Tokens on that card are returned to the supply. (They learned a lot, but only mastered one thing).

**3. Failure (Pivot/Scrap)**
* If the Product is Scrapped (voluntarily or via Audit), **ALL** Pending Tokens (of all colors) are returned to the supply.

### C. MULTI-SKILLED JUNIORS (The Polymath)
Juniors can be trained to handle multiple roles, but they cannot do everything at once.

1.  **Acquiring New Skills:**
    * Play a **Strategy Card** (e.g., "Full Stack Bootcamp") to place a **Skill Token** (Blue, Red, or Green) on the card.
    * *Effect:* The Junior immediately gains **Base Level 1** in that new skill.
2.  **Using the Skills (Mode Switching):**
    * During the **Engine Phase**, you must declare the Junior's **Mode** for the turn (Blue, Red, or QA). They produce only that type.
3.  **Leveling Up (Selective Growth):**
    * XP Tokens are placed on specific slots.
    * *Effect:* The +1 Bonus applies *only* when the Junior is in that specific mode.

> **CRITICAL RESTRICTION: RANK BADGES**
> If you place a **Gold Rank Badge** on a Junior who has the QA Skill:
> * **Effect:** They become a **Tier 2 Lead Developer**. They satisfy the "Lead Requirement" for Tier 2/3 products.
> * **Limitation:** They **DO NOT** gain the "Director of QA" perk (Crisis Management). They **cannot** remove Scandal Tokens. That ability is exclusive to the **QA Specialist** card type.

### D. SPECIALIST GROWTH (QA & Sales)
Specialists do not gain XP from launching. You must use the **Train Action ($3)** to buy them XP Tokens.

1.  **QA Engineer:**
    * **Base:** Removes 1 Bug.
    * **+1 XP:** Removes 2 Bugs.
    * **+2 XP:** Removes 2 Bugs **AND** prevents Revenue Decay.
    * **Rank Path (Promotion):**
        * **Gold Badge:** Becomes **"Director of QA."**
        * **Perk:** Gains **"Crisis Management."** Can remove **Scandal Tokens** (-5 VP) instead of Bugs.
2.  **Sales Rep:**
    * **Base:** +$2 Revenue.
    * **+1 XP:** +$3 Revenue.
    * **+2 XP:** +$4 Revenue.
    * **Rank Badge:** Becomes **VP of Sales**. Ignores Bug Decay ($1/bug penalty is waived).

### E. PRODUCT MANAGERS (The Leadership Track)
PMs do not produce cubes, so they do not use XP Tokens. Instead, they gain **Rank Badges**.

**1. Junior PM (The Scrum Master)**
* **Effect:** **Synergy** (+1 Output per teammate).
* **Promotion Trigger:** Being part of a team that Launches a **Tier 2 or Tier 3 Product**. (Tier 1 does not count).

**2. The Promotion (Pending Rank)**
* **Pending:** If a Junior PM is assigned to a Tier 2/3 Project, place a **Gold Rank Badge (Pending)** on them.
* **Launch:** If successful, the Badge becomes **Permanent**.
* **Scrap:** If failed, the Badge is lost.

**3. Senior PM (Head of Product)**
* **Effect:** A PM with a Gold Badge gains **Agile Leadership**.
* **The Agile Perk:**
    * Any Employee **Reassigned TO** a project managed by a Senior PM does **NOT** receive an Onboarding Token. They work immediately.
    * *Restriction:* This perk applies to the *incoming staff*, not the PM. If the Senior PM themselves moves to a new project, the PM receives an Onboarding Token (disabling their Synergy for that turn).
* *Note:* PMs are Non-Technical. Even with a Gold Badge, they **cannot** fulfill the "Lead Requirement" for Tier 2/3 projects.

### F. SENIORS (The Ceiling)
Cards marked **"Senior"** (Developers) represent hired experts.
* **Static Stats:** They start with High Output (3) but **DO NOT** gain XP Tokens from launching products.
* **Improvement:** The only way to improve a Senior is to attach an **Attribute Card** (e.g., "Visionary") from the Strategy Deck.

### G. THE LEADERSHIP TRACK (Developers Only)
**Rank Badges (Gold)** increase Status/Tier.

**1. Tier 2 (Lead / Senior)**
* **Who is Tier 2?**
    * Any **Senior Developer** (Starts at Tier 2 automatically).
    * Any **Junior Developer** holding **1 Gold Badge** (via Training Action $3).
* **Effect:** Satisfies the **Lead Requirement** for the *specific project* they are assigned to.
* **Poach Defense:** Adds **+$4** to Base Cost (if Junior) or **+$2** (if Senior).

**2. Tier 3 (Executive / CTO)**
* **Limit:** Maximum **1 Tier 3 Talent** per player.
* **Prerequisite:** Must target an existing **Tier 2** unit.
* **The Promotion Cost (The Partnership):**
    * Pay **$5 Cash** to the Bank.
    * **AND** Attach 1 **Equity Token** (from your supply) to the card.
* **Effect A (Tenure):** This unit is **Immune to Poaching**. (The attached Equity acts as a permanent Counter-Offer).
* **Effect B (Audit Shield):** The specific project this unit is assigned to (Dev or Ops) is **Immune to Audits**.
* **Risk:** If you Fire this unit, the attached Equity Token is **Destroyed** (Lost VP).

**3. _The "Rising Star" Rule_: A Junior with a Rank Badge _STILL GAINS XP_ from launches.**
* *Why:* They retain their "Junior" card type. They are simply a Junior with authority.
* *Strategy:* You can build a unit that is **Tier 2 (Lead)** AND has **5 Output** (Max XP). This is the most valuable unit in the game.

---
## 12. SCORING
**Trigger:** Market Crash Card drawn **OR** Turn Limit reached.
* **Market Crash:** Finish the current round, then calculate Final Valuation.
* **Turn Limit (30 Turns):** If 30 complete rounds have been played without the Market Crash card being drawn, the game ends immediately after the current round. Calculate Final Valuation as normal. *(This represents regulatory pressure forcing all startups to demonstrate value.)*

**Valuation Formula:**
1.  **Product Portfolio:** Active Maint Products (+Hype/-Scandal + Integration).
2.  **Cash:** $1 VP per $5.
3.  **Market Share:** 2 VP per Token.
4.  **Portfolio:** 5 VP per Opponent Equity Token.
5.  **Milestones:** Face Value.
6.  **Human Capital:**
    * **1 VP** per XP/Skill Token.
    * **2 VP** per Rank Badge.
7.  **Penalties:**
    * **Vaporware:** Backlog cards are -2/-5/-10 VP.
    * **Debt:** -5 VP per Debt Token.

---

## 13. THE HIRING PROCESS (Detailed)

Acquiring talent is the fuel for your engine. There are three distinct ways to hire, depending on whether you need cheap labor, specific experts, or are digging for hidden gems.

### A. RECRUITING (Standard Action)
This action allows you to hire **visible** cards from the board.

**1. The University (Juniors)**
* **Target:** The infinite stacks of **Junior Software** or **Junior Hardware**.
* **Cost:** **$2** (Paid to Bank).
* **Destination:** **The Bench (Hand)**.
* *Note:* Juniors cannot be placed directly on the board. They must be trained/assigned later.

**2. The Agency Row (Headhunting)**
* **Target:** Any of the 4 face-up cards in the Agency Row (Seniors/Specialists).
* **Cost:** The **Printed Cost** on the card ($4-$8).
* **Destination:**
    * **Option A:** **The Bench** (Safe, free upkeep).
    * **Option B (Immediate Deployment):** You may place the card **Directly onto the Board** (Dev or Ops). This saves you the 1 AP "Assign" action but requires you to pay their salary in the next Income Phase.
* **Refill:** Immediately slide remaining cards to the left and draw a new card from the Talent Deck to fill the empty slot.

**3. The Open Job Market (Bargain Bin)**
* **Target:** Any card in the Open Market queue.
* **Cost:** **$1** (Flat rate, regardless of the card's original value).
* **Destination:** **The Bench (Hand)**.
* *Strategy:* This is where you find cheap Seniors that other players discarded or couldn't afford.

### B. SOURCING (The Gamble)
If the Agency Row doesn't have what you need, you can search for new talent using the **"Layoff / Source"** action.

**The Process:**
1.  **The Sacrifice:** You must discard **X cards** from your Hand to the **Open Job Market**. (X = The number of cards you want to reveal).
    * *Example:* Discard 2 Juniors to look at 2 new cards.
2.  **The Reveal:** Flip the top **X cards** from the **Talent Deck**.
3.  **The Decision:**
    * **Hire:** You may hire one (or more) of the revealed cards by paying their **Printed Cost**.
    * **Pass:** If you cannot afford them (or don't want them), you pass.
4.  **The Fallout:**
    * **Hired Cards:** Go to **The Bench** (or Board via Immediate Deployment rule).
    * **Passed Cards:** These enter the **Agency Row** from the **Left**.
        * Existing Agency cards slide Right.
        * Any card pushed out of the 4th slot falls into the **Open Job Market** (Price drops to $1).

### C. SUMMARY TABLE: RECRUITMENT RULES

| Source | Card Visibility | Cost | Deployment | Strategic Use |
| :--- | :--- | :--- | :--- | :--- |
| **University** | Always Visible | **$2** | **Bench Only** | Cheap labor, XP potential. |
| **Agency** | Visible (4) | **Printed ($4-8)** | **Bench OR Board** | Speed, Specific Roles. |
| **Open Market** | Visible (Max 5) | **$1** | **Bench Only** | Value, Sniping discards. |
| **Talent Deck** | Hidden | **Printed ($4-8)** | **Bench OR Board** | Digging for specific cards. |

---
## 14. THE PRODUCT PIPELINE (Detailed)

Bringing a product to life is a two-step process: **Ideation** (getting the idea) and **Greenlighting** (starting the work). Product cards drawn to the backlog are hidden from other players.

### A. IDEATION (Drafting to Backlog)
**Action Cost:** 1 AP (Standard) or 2 AP (Open Pool).

**1. The Product Market (Standard Draft)**
* **Target:** Any face-up card in the Market Row.
* **Seed Row (Left):** Tier 1 Apps/Devices. Easy to build, low barriers.
* **Growth Row (Right):** Tier 2/3 Enablers and Unicorns. High barriers.
* **Action:** Move the card to your **Backlog (Hand)**.
* **Refill:** Immediately draw a replacement from the corresponding deck.

**2. Research (Blind Draft)**
* **Target:** The top of the Seed Deck OR Growth Deck.
* **Action:** Draw **3 Cards**. Keep **1**.
* **Discard:** Place the other 2 in the **Open Idea Pool**.
* *Strategy:* Use this to dig for specific tags or keep a Unicorn hidden from opponents.

**3. The Open Idea Pool (Recycling)**
* **Target:** Any card in the Open Idea Pool queue.
* **Cost:** **2 AP**. (Higher cost because you are reviving a known, discarded asset).
* **Action:** Move the card to your **Backlog**.

> **Constraint:** Your Backlog (Hand) has a **Limit of 3 Cards**. You cannot Ideate if you have 3 cards, unless you perform an "Overflow Flush" (Draft 4th $\rightarrow$ Discard 1 during Cleanup).


### B. GREENLIGHTING (Starting Development)
**Action Cost:** 0 AP. (This is a free commitment action).

**The Process:**
Move a card from your **Backlog** to your **Development Zone**.

**1. Capacity Check:**
* You must have an open slot in your Dev Zone.
* **Limit:** Maximum **3 Active Projects**. If full, you cannot Greenlight.

**2. Dependency Check (The Gate):**
Does the Product require a Tag (e.g., `[AI]`)?
* **I have the Tag (Owned/Equity):** Project starts for **Free**.
* **I need the Tag:** You must pay a **$3 Licensing Fee** to a player who owns it.
* **No one has the Tag:** You **cannot** Greenlight this project (unless it is Tier 1 with no requirements).
* **Important:** Products in Development do **NOT** provide their tags. Only Launched (Maintenance) products provide tags for licensing and Secured Access.
* **Lead Requirement Warning (Tier 2/3):** Tier 2/3 products require a Matching Lead to generate Output and to Launch. You may Greenlight without a Lead, but the project will be **Stalled (0 Output)** until a Lead is assigned.

**3. Placement Mode:**
* **Public Mode (Face Up):** Standard play. Proof of Tag possession/Licensing is public.
* **Stealth Mode (Face Down):**
    * **Requirement:** You may *only* go Stealth *legally* if you have **Secured Access (Free License):** You own the Tag in your Maintenance Zone OR hold Equity in a player who owns the Tag.
    * **Restriction:** You can go into stealth mode *illegally* meaning you can go to stealth mode even if you don't have **Secured Access** to the tag(s) required by the product.

> **Note:** Once a card is in the Development Zone, it is "Live." You must pay salaries for staff assigned to it.

---
## 15. CARD LIFE CYCLES & FLOW

Understanding how cards move between zones is critical for managing the economy.

### A. THE TALENT ECOSYSTEM
Talent cards represent people. They circulate through the market until they are retired.

**1. Entry (The Source)**
* **University Stacks:** Infinite supply of Juniors.
* **Talent Deck:** Supplies the **Agency Row** (4 Slots). When a card is bought, the row slides left, and a new card enters from the Deck.

**2. Employment (The Player Zones)**
* **Deck/University $\rightarrow$ Bench:** The standard path for new hires.
* **Agency $\rightarrow$ Board:** Seniors can skip the Bench and deploy immediately.
* **Bench $\leftrightarrow$ Board:** Cards move between Hand and Active Duty via Assign/Recall.

**3. The Drain (The Open Job Market)**
* **Hand $\rightarrow$ Open Market:** Occurs when you discard due to Hand Limit (Max 5).
* **Board $\rightarrow$ Open Market:** Occurs when you Fire unpaid staff.
* **Agency $\rightarrow$ Open Market:** Occurs when the Agency Row is full and a player "Passes" during a Source action (The right-most card is bumped).

**4. The Void (Game Removal)**
* The **Open Job Market** is a FIFO Queue (First-In, First-Out) of **5 Cards**.
* When a 6th card enters the Open Market, the **Oldest Card** (at the bottom of the queue) is **Removed from the Game** (returned to box).

### B. THE PRODUCT ECOSYSTEM
Product cards represent Intellectual Property (IP). Even failed projects leave a paper trail.

**1. Entry (The Source)**
* **Seed Deck:** Refills the Left 2 slots of the Market Row.
* **Growth Deck:** Refills the Right 2 slots of the Market Row.

**2. Development (The Player Zones)**
* **Market $\rightarrow$ Backlog:** Drafted cards go to your Hand (Max 3).
* **Backlog $\rightarrow$ Development:** Cards are "Greenlit" to the board.
* **Development $\rightarrow$ Maintenance:** Cards are "Launched" to generate revenue.

**3. The Recycling Bin (Open Idea Pool)**
Cards move to the **Open Idea Pool** (Queue of 5) via:
* **Backlog Overflow:** Discarding due to Hand Limit (>3).
* **The Pivot:** Scrapping an active Development project.
* **Bank Fire Sale:** Selling a Maintenance product to the Bank.

**4. Revival & Obsolescence**
* **Pool $\rightarrow$ Backlog:** Any player can draft from the Pool (Cost: 2 AP).
* **Pool $\rightarrow$ The Void:** If the Pool exceeds 5 cards, the **Oldest Card** is removed from the game (Obsolete).

### C. THE STRATEGY ECOSYSTEM
Strategy cards represent temporary tactics and corporate maneuvers.

* **Deck $\rightarrow$ Hand:** Drawn via the **"Brainstorm"** action.
* **Hand $\rightarrow$ Discard Pile:** Cards enter the discard pile in three ways:
    * **Played:** Instant cards (e.g., "Design Sprint") or Reaction cards (e.g., "Cease & Desist") are discarded after use.
    * **Flushed (Voluntary):** Discarded during the **Brainstorm** action to cycle your hand before drawing new cards.
    * **Cleanup (Forced):** Discarded at the end of the turn if Hand Size exceeds 3.
* **Hand $\rightarrow$ Attachment:** "Attribute" cards are tucked under Talent cards. They stay there until the Talent is Fired/Poached (then Discarded).
* **Reshuffle:** If the Strategy Deck runs out, shuffle the Discard Pile to form a new Deck.

### D. THE EVENT ECOSYSTEM
* **Deck $\rightarrow$ Active Slot:** One card revealed per round.
* **Active Slot $\rightarrow$ Discard:** Replaced at the start of the next round.

---
## 16. THE HOSTILE OFFER (Detailed Poaching Rules)

Poaching allows you to bypass the Market and hire talent directly from an opponent's active workforce. It is expensive, but it acts as a powerful disruption tool.

### A. THE RULES OF ENGAGEMENT
Before calculating cost, you must verify the Poach is legal.

**1. Targeting Restrictions**
* **Valid Targets:** You may only target employees on the **Board** (Development Teams).
* **Invalid Targets:**
    * **The Bench:** Staff here are immune.
    * **The Ops Zone:** Staff here are immune (Golden Handcuffs).
    * **Protected Teams:** Teams with an HR Manager or Investor Immunity.
* **Space Requirement:** You must have an **Immediate Open Slot** on your own Board (Dev or Ops) to place the poached employee. You cannot poach to your Bench.

**2. The Defenses (Automatic Fail)**
The Poach attempt fails immediately if:
* **HR Shield:** The target is on a team assigned with an **HR Manager**.
* **Investor Immunity:** You hold an **Equity Token** of the target player (Partners cannot poach).
* **Vested Interest:** The target card has an **Equity Token** physically placed on it (Counter-Offer).

### B. CALCULATING THE COST
The Poaching Fee is paid to the **BANK** (Headhunter Fees), not the opponent.

**The Formula:**
> **Total Cost = (Base Cost + Token Value) x 2**

**1. Determine Base Cost**
Look at the Printed Cost on the card.
* **Juniors:** $2.
* **Specialists:** $4 - $5.
* **Seniors:** $5 - $8.

**2. Add Token Value (The "Certified" Modifier)**
Add value to the Base Cost for every token physically on the card.
* **Standard Tokens (+$2):** XP Tokens, Skill Tokens, Attributes, and Pending Tokens.
* **Rank Badge on SENIOR (+$2):** A Senior with a Gold Badge adds +$2.
* **Rank Badge on JUNIOR (+$4):** A Junior with a Gold Badge adds **+$4** (The "Growth Premium").

### C. COST EXAMPLES

| Scenario | Base | Tokens | Adjusted Base | **Final Cost (x2)** |
| :--- | :--- | :--- | :--- | :--- |
| **Raw Junior** | $2 | None | $2 | **$4** |
| **Trained Junior** | $2 | 1 Skill Token (+$2) | $4 | **$8** |
| **Super Junior** | $2 | 1 XP (+$2) + Rank (+$4) | $8 | **$16** |
| **Standard Senior** | $6 | None | $6 | **$12** |
| **Buffed Senior** | $6 | 1 Attribute (+$2) | $8 | **$16** |

### D. EXECUTION & RESOLUTION

**1. The Payment**
* The Aggressor pays the Total Cost to the Bank.

**2. The Counter-Offer (Last Chance Defense)**
* The Victim may immediately choose to **Vest** the employee.
* **Action:** Victim takes one of their own **Equity Tokens** and places it on the Talent Card.
* **Result:** The Poach is **Canceled**. The Aggressor takes their money back. The Equity Token is considered "Spent" (Locked for VP, cannot be moved).

**3. The Transfer (If no Counter-Offer)**
* The Aggressor moves the Talent Card (and all attached Tokens/Attributes) to their Board.
* **Dev Team Impact:** If the stolen employee was a **Lead** on a Tier 2/3 project, the project immediately **Stalls** (0 Output) until replaced.
* **Ops Zone Impact:** If the stolen employee causes Supply < Demand, the Victim must immediately choose products to go **Offline**.

---
## 17. THE STOCK MARKET (Strategic Investment)

Investing is the primary way to form alliances and secure passive income. It turns opponents into partners.

### A. BUYING EQUITY (The Investment Round)
* **Action:** **Invest (1 AP)**.
* **Target:** Choose any opponent.
* **Cost:** Pay cash to that opponent based on their **Company Stage**. You cannot negotiate the price; it is fixed by the market.

| Target's Highest Active Product | Share Price (Cost) |
| :--- | :--- |
| **Seed Stage** (Tier 1 Only) | **$5** |
| **Growth Stage** (Has Tier 2) | **$10** |
| **Unicorn Stage** (Has Tier 3) | **$15** |

* **Effect:** Take 1 **Equity Token** of the opponent's color and place it Face-Up in your play area.

### B. THE TERM SHEET (Investor Perks)
As long as you hold an opponent's Equity Token, you gain the following 5 benefits. These effects apply immediately.

**1. Dividends (Passive Income)**
* During the **Income Phase**, when the Founder (opponent) calculates their **Operational Revenue**, the **BANK** pays you a dividend.
    * **$0 Revenue:** $0 Dividend.
    * **$1 - $10 Revenue:** $1 Dividend.
    * **$11 - $20 Revenue:** $2 Dividend.
    * **$21+ Revenue:** $3 Dividend.

**2. Non-Aggression Pact (Immunity)**
* You (the Investor) **cannot** play "Headhunter" cards or "Hit Piece" cards against this Founder.
* *Note:* The Founder is **NOT** immune to you. They can still attack you (unless they buy your Equity back).

**3. Strategic Partnership (Free Licensing)**
* **Reciprocal Access:** BOTH you and the Founder may use each other's **Maintenance Tags** (`[AI]`, `[Platform]`, etc.) to satisfy Draft/Greenlight requirements.
* **Cost:** **$0**. (Do not pay the $3 License Fee).
* **Stealth Rights:** Holding Equity counts as **"Secured Access,"** allowing you to draft Face-Down relying on their tags.

**4. The Consultant (Active Buff)**
* **Placement:** You may physically place the Equity Token onto **ONE** of the Founder's active Board Cards (Dev or Ops).
* **Effect:** That card gains **+1 Output** (Blue/Red/QA - your choice) during the Engine Phase.
* **Mobility:** At the start of every Round (Event Phase), you may move the token to a different card on the Founder's board.
* *Strategy:* Buff their highest revenue product so they hit the next Dividend Tier.

**5. The Veto (The Nuclear Option)**
* **Trigger:** You may interrupt the Founder's turn when they declare an action.
* **Cost:** **Burn** (Discard) the Equity Token back to the Founder. The founder may resell the token to a new investor later.
* **Effect:** The Founder's action is **Canceled**. The AP is wasted.
* *Result:* The Alliance is broken. You lose all other perks immediately.

### C. LOSING & MOVING EQUITY (Divestment)
You lose the token (and all perks) if:
1.  **You use the Veto.**
2.  **Non-Compete Lawsuit (Hostile Buyout):**
    * The Founder may play the **"Non-Compete Suit"** Strategy Card.
    * **Effect:** The Founder pays you the **Seed Price ($5)**.
    * **Result:** You **MUST** return the Equity Token to the Founder immediately. You lose the VP and all future dividends.
3.  **Bankruptcy:** If you (Investor) go bankrupt, your assets are seized and returned to the founder.
4.  **The Buyback (Transaction):**
    * **Voluntary Buyback (1 AP):** The Founder offers to buy their token back at the **Current Share Price** ($5/$10/$15). If you agree, you get the cash and return the token.
5.  **Secondary Trade (Investor-to-Investor):**
    * An Investor may sell their stake to a third party.
    * **Action:** whoever initiates the trade spends **1 AP**.
    * **Price:** Current Share Price ($5/$10/$15).
    * **Consent:** both parties must agree. **The Founder cannot prevent this sale.**
    * *Effect:* The Token (and all Perks) transfer to the new owner immediately.


### D. Investment Limits (Controlling Interest)
* **Max Ownership:** You may buy multiple Equity Tokens from the same player.
* **The Hard Cap:** A player must always retain **1 Equity Token** of their own color. They cannot sell their last token.
    * *Result:* There are only **2 Tokens** available for purchase per player. First come, first served.
* **Stacking Benefits:**
    * **Dividends:** Cumulative (e.g., holding 2 tokens = 2x Dividend payment).
    * **Veto:** You may use tokens individually. Burning one leaves the other active (preserving your Alliance status).
    * **Consultant:** You may place **both** tokens on the Founder's board to give **+2 Output** (or +1 to two different cards).

## 18. MANAGING BOARD TALENT (Firing & Recycling)

Players often want to remove staff from the Board, either to save money or to use them for Sourcing. It is critical to distinguish between **Firing** (Removal) and **Laying Off** (Sourcing).

### A. THE RESTRICTION
You **CANNOT** use staff currently on the **Board** (Dev Teams or Ops Zone) to pay the cost for the **"Layoff / Source"** action.
* **Rule:** The "Source" action specifically requires discarding from your **Bench (Hand)**.
* *Why:* Staff on the board are "busy." You cannot send them to the job market to find their own replacement unless you pull them off the line first.

### B. DISMISSAL (Firing from Board)
If you want to remove staff from the board *without* returning them to your hand, you may **Dismiss** them.
* **Timing:**
    * **Income Phase:** Simply refuse to pay their salary.
    * **Action Phase:** Voluntarily discard them to the **Open Job Market** to clear a slot for a new hire or poach.
* **Effect:** The staff member leaves immediately. You pay **$0** severance.
* **Result:** You **DO NOT** draw new cards. This is a pure removal action.

### C. THE RECYCLING LOOP (Board -> Bench -> Source)
If you want to use an active Board employee to dig for new talent (Source), you must execute a **2-Step Maneuver**:

1.  **Step 1: Recall (1 AP)**
    * Move the staff member from **Ops Zone** $\rightarrow$ **Bench**.
    * *(Note: If they are in Development, you must **Pivot** (Scrap Project) to get them back).*
2.  **Step 2: Source (1 AP)**
    * Now that they are on the **Bench**, you may discard them to trigger the **Source** action (Draw cards from Deck).

> **Summary:**
> * **Board $\rightarrow$ Open Market:** Free (Dismissal). No benefit.
> * **Board $\rightarrow$ Source:** Impossible directly. Must Recall first (Cost 2 AP total).

## 19. SECTOR SYNERGY & TECH TREES

To succeed in the mid-game, players are encouraged to specialize in specific industries. This is represented by the **Domain Expertise** mechanic.

### A. THE DOMAIN EXPERTISE RULE (Synergy Bonus)
If you have at least one **Active Product** in your **Maintenance Zone** belonging to a specific Industry Sector (Pink, Gold, Gray, or Green), you gain momentum in that field.

* **The Bonus:** Future products you Draft/Greenlight from that **SAME SECTOR** cost **-2 Work Cubes** to develop. (player can choose between -2 {}, -2 [Chk], -1 {} and -1 [Chk] for hybrid products)
    * *Minimum Cost:* A product's cost cannot be reduced below **1 Cube**. (Combined cost of {} and [Chk] cannot be less than 1)
    * *Restriction:* This bonus applies to **Blue `{}`** and **Red `[Chk]`** costs equally. (e.g., A cost of 6 Blue becomes 4 Blue).
* **Infrastructure Exception:** The **Blue Sector (Infrastructure)** does **NOT** generate or receive Synergy Bonuses. These are universal utility cards.

### B. THE 5 SECTORS (Detailed)

#### 1. 📱 CONSUMER & MEDIA (Pink)
* **Playstyle:** High Revenue ($), High Churn (Bugs).
* **Synergy:** Active Pink Product = -2 Cost on future Pink cards.
* **Tech Tree:**
    * **Tier 1:** `[Social]` (Dating Apps, Viral Video).
    * **Tier 2:** `[Media]` (Streaming, Esports). *Requires `[Social]`.*
    * **Tier 3:** `[Metaverse]` (Virtual World). *Requires `[Media]` + `[Cloud]`.*

#### 2. 💰 FINTECH (Gold)
* **Playstyle:** Cash Flow & Licensing Dominance.
* **Synergy:** Active Gold Product = -2 Cost on future Gold cards.
* **Tech Tree:**
    * **Tier 1:** `[Commerce]` (Wallets, Gateways).
    * **Tier 2:** `[Crypto]` (Neobanks, NFTs). *Requires `[Commerce]`.*
    * **Tier 3:** `[DeFi]` (Global Currency). *Requires `[Crypto]` + `[Cloud]`.*

#### 3. ⚙️ DEEP TECH (Gray)
* **Playstyle:** Hardware Heavy (`[Chk]`). Slow build, High Stability.
* **Synergy:** Active Gray Product = -2 Cost on future Gray cards.
* **Tech Tree:**
    * **Tier 1:** `[IoT]` (Smart Devices, Wearables).
    * **Tier 2:** `[Robotics]` (Drones, Automation). *Requires `[IoT]`.*
    * **Tier 3:** `[Fusion]` (Energy, Mars Colony). *Requires `[Robotics]` + `[AI]`.*

#### 4. 🧬 LIFE SCIENCE (Green)
* **Playstyle:** The Long Game. High Cost, Massive Valuation (VP).
* **Synergy:** Active Green Product = -2 Cost on future Green cards.
* **Tech Tree:**
    * **Tier 1:** `[Data]` (Health Trackers).
    * **Tier 2:** `[Bio]` (Diagnostics, Lab Meat). *Requires `[Data]`.*
    * **Tier 3:** `[Longevity]` (Immortality). *Requires `[Bio]` + `[AI]`.*

#### 5. ☁️ INFRASTRUCTURE (Blue)
* **Playstyle:** The Backbone. Essential dependencies for all other sectors.
* **Synergy:** **None.** (Pay full price).
* **Tech Tree:**
    * **Tier 1:** `[Platform]` (OS, Blogs).
    * **Tier 2:**
        * `[Cloud]` (Server Farms). *Required for Pink/Gold Tier 3.*
        * `[AI]` (Search Engines). *Required for Gray/Green Tier 3.*
    * **Tier 3:** `[Quantum]` (Computing). *The ultimate calculator.*

## 20: THE CARD & COMPONENT MANIFEST (v5.3)

### A. Strategy Deck (30 Cards)
* **Training (8):**
    * 2x **Full Stack Bootcamp** ($3).
    * 2x **QA Certification** ($3).
    * 2x **Sales Seminar** ($2).
    * 1x **Leadership Masterclass** ($3) *(Renamed from Agile Workshop)*.
    * 1x **Agile Workshop** ($3): Add **Gold Rank Badge** to PM (Becomes Agile).
* **Warfare (10 - Increased):**
    * **4x Headhunter:**
        * *Effect:* Poach target for **2x Cost**.
        * *Cost to Play:* $0.
    * **2x Corporate Raider (Rare):**
        * *Effect:* Poach target for **1.5x Cost** AND bypass HR Manager.
        * *Cost to Play:* $2.
    * 2x **Hit Piece** ($3): Add Scandal.
    * 2x **Non-Compete Suit** ($4): **Hostile Buyout.** Pay Investor **$5** to force return of Equity.
* **Attributes (6):**
    * 1x **Workaholic** ($2): Buff Own (+2 Output, +$2 Sal).
    * 1x **Clean Coder** ($3): Buff Own (This unit never generates Bugs).
    * 1x **Visionary** ($3): Buff Own (Counts as Lead for any Tier).
    * 1x **Toxic Personality** ($3): Debuff Enemy (-1 Team Output).
    * 1x **Burnout** ($0): Debuff Enemy (0 Output).
    * 1x **Flight Risk** ($2): Debuff Enemy (Poach Cost is 1x Base).
* **Utility (6):**
    * 2x **TechCrunch Feature** ($5): Add Hype.
    * 2x **Design Sprint** ($0): Draw 5 Products.
    * 2x **Cease & Desist** ($0): Cancel Attack.

### B. Talent Deck (22 Cards)

#### 1. THE AGENCY DECK (Seniors & Specialists)
* **Total Cards:** 22
* **Card Back:** "Agency Talent"
* **General Rule:** All cards from this deck are eligible for **Immediate Deployment** (can be placed directly onto the Board, skipping the Bench).

##### A. SENIOR DEVELOPERS (10 Cards)
*High output but expensive. They come with "Baggage" (Traits).*

| Count | Role / Name | Cost | Salary | Output | Trait / Effect |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **1x** | **Senior Backend** (The Hacker) | **$6** | **$2** | **3 `{}`** | **Spaghetti Code:** On entry, immediately add **1 Bug Token** to the product. |
| **1x** | **Senior Backend** (The Architect) | **$7** | **$2** | **3 `{}`** | **Clean Code:** This unit **never** generates Bug Tokens (ignores Growth Hacker/Event effects). |
| **1x** | **Senior Backend** (The Veteran) | **$6** | **$2** | **3 `{}`** | **Mentor:** When a Junior on this team Launches, they gain **+1 extra XP**. |
| **1x** | **Senior Hardware** (The Diva) | **$5** | **$1** | **3 `[Chk]`** | **Ego:** This card **cannot be Reassigned** to another project. It must stay until Launch or be Fired. |
| **1x** | **Senior Hardware** (The Engineer) | **$6** | **$2** | **3 `[Chk]`** | **Efficient:** Salary is **$1** if in Ops Zone. |
| **1x** | **Senior Hardware** (The Fixer) | **$7** | **$2** | **3 `[Chk]`** | **QA Skill:** Owner may switch this unit's mode to **Remove 1 Bug** instead of producing cubes. |
| **2x** | **Firmware Specialist** | **$6** | **$2** | **2 (Flex)** | **Versatile:** Counts as **Blue `{}` OR Red `[Chk]`**. Owner declares mode each turn. |
| **2x** | **Full Stack Ninja** | **$8** | **$3** | **3 (Flex)** | **Mercenary:** The Poaching Cost for this unit is **1.5x Base Cost** (instead of 2x). |

##### B. SPECIALISTS (12 Cards)
*Utility roles that do not produce work cubes but provide critical buffs.*

| Count | Role | Cost | Salary | Output | Trait / Effect |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **3x** | **QA Engineer** | **$4** | **$1** | **0** | **Bug Hunter:** <br>• **Dev:** Remove **1 Bug** from current project per turn.<br>• **Ops:** Remove **1 Bug** from any Maintenance product per turn. |
| **3x** | **Sales Rep** | **$4** | **$0** | **0** | **Rainmaker:** <br>• **Ops:** Attached Product generates **+$2 Revenue**.<br>• **Perk:** PR Campaigns on this product cost **$2** (Half Price). |
| **2x** | **HR Manager** | **$5** | **$1** | **0** | **Gatekeeper:** <br>• **Passive:** Opponents **cannot Poach** any staff assigned to this specific team/project. |
| **2x** | **Junior PM** | **$5** | **$1** | **0** | **Synergy:** <br>• **Dev:** Generates **+1 Output** (matching color) for every *other* employee on this team.<br>• *Note:* Is Tier 1 (Cannot Lead). |
| **1x** | **Senior PM** (Head of Product) | **$8** | **$3** | **0** | **Agile Synergy:** <br>• **Dev:** Same Synergy effect as Junior PM.<br>• **Agile:** No **Onboarding Penalty** when staff are Reassigned *TO* this team. |
| **1x** | **Growth Hacker** | **$5** | **$2** | **0** | **Viral Loop:** <br>• **Ops:** Attached Product generates **+$3 Revenue**.<br>• **Decay:** You MUST add **1 Bug Token** to this product at the end of every turn. |

#### 2. THE UNIVERSITY STACKS (Juniors)
* **Total Cards:** Infinite (Use ~10-15 of each for prototype).
* **Card Back:** "University" (or same as Agency if you want to allow mixing later, but usually kept separate).
* **General Rule:** Must go to **Bench** when recruited.

##### A. JUNIOR SOFTWARE DEV
* **Cost:** **$2**
* **Salary:** **$0** (Becomes **$1** if card has 2+ XP Tokens).
* **Base Output:** **1 `{}` (Blue)**.
* **Potential:**
    * Includes **4 Empty Circles** for XP Tokens.
    * Max Output: **5**.
    * Can acquire **Red/Green Skill Tokens** via Training.

##### B. JUNIOR HARDWARE ENG
* **Cost:** **$2**
* **Salary:** **$0** (Becomes **$1** if card has 2+ XP Tokens).
* **Base Output:** **1 `[Chk]` (Red)**.
* **Potential:**
    * Includes **4 Empty Circles** for XP Tokens.
    * Max Output: **5**.
    * Can acquire **Blue/Green Skill Tokens** via Training.

### C. Product Deck (30 Cards)
*Cost: Blue {} / Red [Chk]. Maint: Bandwidth Cost.*

#### 1. THE SEED DECK (Tier 1) - 20 Cards
*Refills Left Market Slots. No Prerequisites.*

| Sector | Color | Card Name | Cost | Rev | Val | Maint | Tag |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Concept** | White | **To-Do List** | 4 `{}` | $1 | 1 | 1 `{}` | `[App]` |
| **Concept** | White | **Flashlight App** | 4 `{}` | $1 | 1 | 1 `{}` | `[App]` |
| **Concept** | White | **Basic Website** | 4 `{}` | $1 | 1 | 1 `{}` | `[Platform]` |
| **Concept** | White | **USB Gadget** | 4 `{}` | $1 | 1 | 1 `{}` | `[Device]` |
| **Consumer** | Pink | **Dating App** | 6 `{}` | $1 | 5 | 1 `{}` | `[Social]` |
| **Consumer** | Pink | **Viral Video App** | 6 `{}` | $2 | 3 | 1 `{}` | `[Social]` |
| **Consumer** | Pink | **Messaging App** | 4 `{}` | $1 | 2 | 1 `{}` | `[Social]` |
| **FinTech** | Gold | **Mobile Wallet** | 4 `{}` | $2 | 2 | 1 `{}` | `[Commerce]` |
| **FinTech** | Gold | **Payment Gateway** | 6 `{}` | $3 | 2 | 1 `{}` | `[Commerce]` |
| **FinTech** | Gold | **Trading Bot** | 6 `{}` | $4 | 1 | 2 `{}` | `[Commerce]` |
| **Deep Tech**| Gray | **Smart Thermostat** | 2`{}`/4`[Chk]` | $2 | 3 | 1 `[Chk]` | `[IoT]` |
| **Deep Tech**| Gray | **Fitness Wearable** | 2`{}`/4`[Chk]` | $2 | 3 | 1 `[Chk]` | `[IoT]` |
| **Deep Tech**| Gray | **Smart Lock** | 1`{}`/5`[Chk]` | $2 | 3 | 1 `[Chk]` | `[IoT]` |
| **Life Sci** | Green| **Health Tracker** | 4 `{}` | $1 | 2 | 1 `{}` | `[Data]` |
| **Life Sci** | Green| **Genetics Kit** | 8 `[Chk]` | $3 | 4 | 1 `[Chk]` | `[Data]` |
| **Life Sci** | Green| **Telehealth Platform**| 6 `{}` | $2 | 3 | 1 `{}` | `[Data]` |
| **Infra** | Blue | **Tech Blog** | 4 `{}` | $1 | 1 | 1 `{}` | `[Platform]` |
| **Infra** | Blue | **Cloud Storage** | 4`{}`/4`[Chk]` | $3 | 2 | 1 `[Chk]` | `[Platform]` |
| **Infra** | Blue | **Gig Economy** | 8 `{}` | $3 | 2 | 1 `{}` | `[Platform]` |
| **Infra** | Blue | **App Store** | 6 `{}` | $4 | 3 | 1 `{}` | `[Platform]` |

#### 2. THE GROWTH DECK (Tier 2 & 3) - 21 Cards
*Refills Right Market Slots. All cards here require a **Lead Talent** to build.*

##### A. TIER 2 PRODUCTS (15 Cards)
*3 Cards per Sector. Mix of Revenue Engines and Tech Enablers.*

| Sector | Card Name | Cost (Dev) | Rev | Val | Maint | **Requires** | **Tag Provided** |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Consumer** | **Streaming Service** | 10 `{}` | $6 | 6 | 2 `{}` | `[Social]` | `[Media]` |
| **Consumer** | **Esports League** | 10 `{}` | $5 | 8 | 2 `{}` | `[Social]` | `[Media]` |
| **Consumer** | **Social VR Space** | 12 `{}` | $4 | 8 | 2 `{}` | `[Social]` | `[Media]` |
| **FinTech** | **Neobank** | 10 `{}` | $3 | 10 | 1 `{}` | `[Commerce]` | `[Crypto]` |
| **FinTech** | **NFT Marketplace** | 10 `{}` | $5 | 6 | 2 `{}` | `[Platform]` | `[Crypto]` |
| **FinTech** | **P2P Lending** | 12 `{}` | $6 | 5 | 2 `{}` | `[Commerce]` | `[Crypto]` |
| **Deep Tech**| **Drone Delivery** | 6`{}`/6`[Chk]` | $5 | 8 | 1`{}`+1`[Chk]`| `[IoT]` | `[Robotics]` |
| **Deep Tech**| **Smart Grid** | 2`{}`/10`[Chk]` | $7 | 5 | 2 `[Chk]` | `[IoT]` | `[Energy]` |
| **Deep Tech**| **Auto Warehouse** | 4`{}`/8`[Chk]` | $6 | 6 | 2 `[Chk]` | `[IoT]` | `[Robotics]` |
| **Life Sci** | **AI Diagnostician** | 14 `{}` | $2 | 12 | 2 `{}` | `[Data]` | `[Bio]` |
| **Life Sci** | **Lab-Grown Meat** | 8`{}`/6`[Chk]` | $4 | 8 | 1`{}`+1`[Chk]`| `[Data]` | `[Bio]` |
| **Life Sci** | **Smart Prosthetics** | 4`{}`/8`[Chk]` | $5 | 8 | 1`{}`+1`[Chk]`| `[Data]` | `[Bio]` |
| **Infra** | **Server Farm** | 8 `[Chk]` | $7 | 4 | 2 `[Chk]` | `[Platform]` | `[Cloud]` |
| **Infra** | **Search Engine** | 12 `{}` | $6 | 8 | 2 `{}` | `[Platform]` | `[AI]` |
| **Infra** | **Cybersecurity Firm**| 14 `{}` | $8 | 4 | 2 `{}` | `[Platform]` | `[Cloud]` |

##### B. TIER 3 UNICORNS (5 Cards)
*1 Card per Sector. Massive VP. Strict Dependencies.*

| Sector | Card Name | Cost (Dev) | Rev | Val | Maint | **Dependency** |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Consumer** | **The Metaverse** | 16 `{}` | $0 | 25 | 3 `{}` | `[Media]` + `[Cloud]` |
| **FinTech** | **Global Currency** | 18 `{}` | $2 | 22 | 2 `{}` | `[Crypto]` + `[Cloud]` |
| **Deep Tech**| **Fusion Reactor** | 4`{}`/20`[Chk]` | $10 | 20 | 3 `[Chk]` | `[Energy]` + `[Quantum]`|
| **Life Sci** | **Immortality Pill** | 10`{}`/10`[Chk]`| $15 | 25 | 2`{}`+2`[Chk]`| `[Bio]` + `[AI]` |
| **Infra** | **Quantum Computer**| 18 `[Chk]` | $10 | 15 | 3 `[Chk]` | `[AI]` *(Prov. Quantum)* |

##### C. Game Over Card (The Market Crash)
| Card Name | Effect |
| :--- | :--- |
| **Market Crash** | **GAME OVER.** Finish the current round. Calculate Final Valuation immediately. |

### D. Event Deck (18 Cards)
*Shuffle and reveal 1 per round. Allows for ~18 rounds of unique gameplay.*

**A. ECONOMIC EVENTS (Cash Flow)**
1.  **Venture Capital Rain:** Equity Sales grant **+$3 extra cash** this round.
2.  **Tech Bubble:** Market confidence is high. All Active Maintenance Products generate **+$1 Revenue** this round.
3.  **Patent Trolls:** IP lawsuits spike. The **Licensing Fee** is increased to **$5** (from $3) this round.
4.  **Angel Investors:** Small startups get a boost. Any player with **Tier 1 Products only** (no Tier 2/3 on board) gains **$3 Cash** immediately.
5.  **Legacy Code:** Technical debt is expensive. **Revenue Decay** is increased to **-$2 per Bug** (instead of -$1) this round.
6.  **Regulation Audit:** **Payroll Tax.** Pay **$1** for every Employee on your Board with a **Salary > $0**. (Juniors with 0-1 XP are exempt). Discard unpaid staff.

**B. PRODUCTION EVENTS (Output & Bandwidth)**
7.  **Open Source Boom:** All Software `{}` output is **+1** this round.
8.  **Hardware Hackathon:** All Hardware `[Chk]` output is **+1** this round.
9.  **Chip Shortage:** Hardware `[Chk]` hiring costs **+$3** this round.
10. **Crunch Time:** Management demands results. All Dev Teams produce **+1 Cube** (Blue or Red), but **MUST** add **1 Bug Token** to the project.
11. **Server Crash:** Cloud infrastructure fails. All **Software `{}` Staff** in the **Ops Zone** produce **-1 Bandwidth** this round. (Check supply vs. demand carefully!).
12. **Supply Chain Logjam:** Hardware components are stuck in customs. All **Hardware `[Chk]`** output is **-1** this round (Dev and Ops).

**C. HR & MANAGEMENT EVENTS (Talent & Training)**
13. **Recruiter Frenzy:** Poaching costs **1x Base Cost** (instead of 2x) this round.
14. **Intern Season:** Fresh graduates flood the market. Recruiting from **University** costs **$0** this round.
15. **Headhunter Holiday:** Recruiters are on break. **Hostile Offers (Poaching)** are **Suspended/Illegal** this round.
16. **Online Course Boom:** Training is accessible. The **Train** action costs **$1** (instead of $3) this round.

**D. CONFLICT & AUDIT EVENTS**
17. **Whistleblower Protection Act:** The government encourages reporting. The **Bank Reward** for a successful Audit (Settlement) is doubled to **$8**.
18. **Trade Show:** Hype is cheap. **PR Campaigns** cost **$2** (Half Price) this round.

### E. OPS ZONE KIT (The Server Rack)
*Instead of placing tokens on cards, players use physical strips inserted into a "Capacity Ruler" on their player mat.*

**1. Components to Make:**
* **Blue Strips `{}`:** Cut form Blue cardstock.
* **Red Strips `[Chk]`:** Cut from Red cardstock.
* **Width:** Uniform (e.g., 2cm).
* **Length:** Represents Maintenance Cost (Bandwidth).
    * **Size 1:** 1 inch long.
    * **Size 2:** 2 inches long.
    * **Size 3:** 3 inches long.

**2. Quantity Checklist:**
* **Blue Size 1 (x12):** Write names like "App", "Blog", "Tracker" on the edge.
* **Blue Size 2 (x6):** Write "Ad Network", "Search Engine", "Consulting".
* **Blue Size 3 (x4):** Write "Metaverse", "Global Currency".
* **Red Size 1 (x8):** Write "Tracker", "Bulb", "Lock".
* **Red Size 2 (x6):** Write "Server Farm", "5G", "Bitcoin".
* **Red Size 3 (x2):** Write "Quantum Computer".

**3. Usage Rule:**
When a player Launches (or Acquires) a product, hand them the specific strip(s) matching that product's Maintenance Cost. They slot it into their Rack immediately.

### F. Expansion Products

#### 1. SEED DECK EXPANSION (Tier 1)
*Refills Left Market. No Prerequisites.*

| Sector | Color | Card Name | Cost (Dev) | Rev | Val | Maint | **Requires** | **Tag Provided** |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Consumer** | Pink | **Indie Game Hit** | 6 `{}` | $3 | 1 | 1 `{}` | None | `[App]` |
| **Consumer** | Pink | **News Aggregator** | 4 `{}` | $1 | 3 | 1 `{}` | None | `[Platform]` |
| **FinTech** | Gold | **Crowdfunding Page**| 4 `{}` | $1 | 3 | 1 `{}` | None | `[Platform]` |
| **FinTech** | Gold | **Budgeting App** | 4 `{}` | $2 | 1 | 1 `{}` | None | `[App]` |
| **Deep Tech**| Gray | **Smart Toy** | 1`{}`/3`[Chk]`| $2 | 2 | 1 `[Chk]` | None | `[Device]` |
| **Deep Tech**| Gray | **Pet Tracker** | 2`{}`/4`[Chk]`| $2 | 3 | 1 `[Chk]` | None | `[IoT]` |
| **Life Sci** | Green | **Sleep Monitor** | 2`{}`/4`[Chk]`| $1 | 4 | 1 `[Chk]` | None | `[Data]` |
| **Life Sci** | Green | **Meditation App** | 4 `{}` | $1 | 2 | 1 `{}` | None | `[Data]` |
| **Infra** | Blue | **VPN Service** | 4 `{}` | $2 | 2 | 1 `{}` | None | `[Platform]` |
| **Infra** | Blue | **Dev Tools** | 6 `{}` | $3 | 2 | 1 `{}` | None | `[Platform]` |

#### 2. GROWTH DECK EXPANSION (Tier 2)
*Refills Right Market. Requires Lead Talent.*

| Sector | Color | Card Name | Cost (Dev) | Rev | Val | Maint | **Requires** | **Tag Provided** |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Consumer** | Pink | **Influencer Agency**| 10 `{}` | $7 | 5 | 2 `{}` | `[Social]` | `[Media]` |
| **Consumer** | Pink | **Music API** | 12 `{}` | $4 | 8 | 2 `{}` | `[Platform]` | `[Media]` |
| **FinTech** | Gold | **InsurTech Platform**| 12 `{}` | $5 | 7 | 2 `{}` | `[Data]` | `[Crypto]` |
| **FinTech** | Gold | **Algo-Trading Firm**| 14 `{}` | $9 | 2 | 2 `{}` | `[Commerce]`| `[Crypto]` |
| **Deep Tech**| Gray | **Vertical Farm** | 4`{}`/10`[Chk]`| $6 | 6 | 2 `[Chk]` | `[IoT]` | `[Robotics]` |
| **Deep Tech**| Gray | **Exoskeleton Suit** | 6`{}`/8`[Chk]` | $5 | 8 | 1`{}`+1`[Chk]`| `[Device]` | `[Robotics]` |
| **Life Sci** | Green | **Genome Editor** | 15 `{}` | $3 | 12 | 2 `{}` | `[Data]` | `[Bio]` |
| **Life Sci** | Green | **Bio-Fuel Cell** | 2`{}`/12`[Chk]`| $6 | 6 | 2 `[Chk]` | `[IoT]` | `[Energy]` |
| **Infra** | Blue | **Blockchain Ledger**| 10 `{}` | $4 | 6 | 2 `{}` | `[Platform]` | `[Cloud]` |
| **Infra** | Blue | **Quantum Prototype**| 2`{}`/12`[Chk]`| $1 | 10 | 2 `[Chk]` | `[Platform]` | `[AI]` |

#### 3. ALTERNATIVE UNICORNS (Tier 3)
*Refills Right Market. Massive VP. Strict Dependencies.*

| Sector | Color | Card Name | Cost (Dev) | Rev | Val | Maint | **Requires** | **Tag Provided** |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Consumer** | Pink | **Neural Social Grid**| 18 `{}` | $2 | 24 | 3 `{}` | `[Media]`+`[Bio]` | -- |
| **FinTech** | Gold | **DAO Nation** | 20 `{}` | $0 | 28 | 2 `{}` | `[Crypto]`+`[Social]`| -- |
| **Deep Tech**| Gray | **Asteroid Mining** | 8`{}`/18`[Chk]` | $12| 18 | 3 `[Chk]` | `[Robotics]`+`[Cloud]`| -- |
| **Infra** | Blue | **Planetary Wi-Fi** | 6`{}`/16`[Chk]` | $8 | 16 | 3 `[Chk]` | `[Cloud]`+`[IoT]` | -- |
