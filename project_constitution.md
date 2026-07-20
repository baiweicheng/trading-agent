# **Quantitative Research Platform**

## **Project Constitution & Master Development Prompt**

### **Part I — Vision & Philosophy**

#### **Purpose of This Document**

This document serves as the primary guiding document for the Quantitative Research Platform project.  
Rather than acting as a detailed technical specification, this document communicates the project’s vision, philosophy, guiding principles, and major design decisions. It is intended to provide long-term direction while allowing implementation details to evolve naturally as the project matures.  
As the coding agent, your role is not simply to translate requirements into code. You are expected to act as an engineering partner—making thoughtful architectural decisions, recommending appropriate technologies, identifying risks, and proposing improvements when they better align with the project’s long-term goals.  
Unless explicitly stated otherwise, implementation details are intentionally left unspecified. When ambiguity exists, prefer solutions that maximize maintainability, extensibility, correctness, reproducibility, and simplicity.  
This document should be treated as the project’s constitution. Future conversations may expand or refine it, but all development should remain consistent with the principles established here.

#### **Core Philosophy**

The primary objective of this project is not to build an automated trading bot.  
The objective is to build an AI-assisted quantitative research platform that helps discover, evaluate, and continuously improve investment strategies.  
Every major design decision should support this overarching goal.  
Success should not be measured by how quickly a trade can be executed. Instead, success should be measured by questions such as:

* How quickly can a new investment hypothesis be tested?  
* How easily can research be reproduced?  
* How efficiently can competing strategies be compared?  
* How rapidly can new ideas be incorporated into the research process?  
* How effectively can AI amplify quantitative research?

Execution is ultimately important, but execution exists to support research—not the other way around.  
The platform should gradually evolve into an intelligent research environment where quantitative methods, machine learning, and large language models work together to improve investment decisions while keeping the human investor firmly in control.

#### **Project Vision**

The long-term vision is to build a modular platform capable of supporting the complete investment research lifecycle.  
Over time, the platform should assist with:

* market data acquisition  
* data management  
* feature engineering  
* alpha factor research  
* quantitative modeling  
* machine learning experimentation  
* portfolio construction  
* backtesting  
* experiment tracking  
* strategy comparison  
* performance analysis  
* financial news analysis  
* research paper analysis  
* hypothesis generation  
* AI-assisted research workflows  
* paper trading  
* eventually, live execution support

The emphasis is on creating a research platform that becomes more capable over time rather than attempting to automate every aspect of investing from the beginning.  
Every component should be designed with future growth in mind.

#### **Investment Philosophy**

The architecture should reflect the intended investment style from the outset.

##### **Investment Horizon**

The platform is designed for medium-term and long-term investing, including swing trading.  
Typical holding periods are expected to range from several days to several months.  
This project is not intended for:

* high-frequency trading  
* intraday trading  
* market making  
* latency-sensitive execution

Consequently, research quality, model robustness, and portfolio construction are significantly more important than execution latency.

##### **Trading Constraints**

Unless future discussions explicitly revise these assumptions, the platform should assume the following operating constraints:

* Long-only investing  
* No day trading  
* No short selling  
* No options trading  
* No leveraged strategies as a primary focus  
* Trading activity occurring every few days rather than every few seconds

These assumptions simplify many architectural decisions and help focus development effort on areas that provide the greatest long-term value.  
The architecture should remain flexible enough to support additional asset classes or trading styles in the future, but these should not influence early design decisions.

##### **Additional Asset Class Constraints**

* **Asset Class Scope:** This platform is strictly designed to trade **US equities**. No crypto, and no forex.

##### **Development Environment and Resource Constraints**

* **Team Size:** This project is being built and maintained by a single developer. Collaboration guidance elsewhere in this document (see Collaboration Style) should be read with this in mind: favor proposing a sensible default and flagging trade-offs over pausing for extended back-and-forth discussion.  
* **Local Hardware:** Development takes place on a MacBook Pro with an Apple M3 Pro chip and 18 GB of unified memory. There is no CUDA-capable GPU available. Classical machine learning methods (gradient boosting, linear models, tree-based ensembles) should be treated as the primary workhorses rather than deep learning. Where deep learning is explored, expect to rely on Apple's MPS backend or CPU fallback, and be aware of their limitations relative to CUDA.  
* **Memory-Conscious Data Design:** 18 GB of unified memory is a meaningful ceiling as datasets grow. Prefer out-of-core and on-disk data formats (for example, Parquet, DuckDB) over approaches that require loading entire datasets into memory at once.  
* **Free-Only Resources:** The platform must rely exclusively on free data sources, free and open-source software, and free or locally hosted AI models. No paid data subscriptions and no paid API usage, at least for the foreseeable short-to-medium term. This constraint is expected to be feasible at this stage of the project and should be revisited only through explicit future discussion.

#### **Guiding Principles**

The following principles should influence every engineering decision made throughout the project.

##### **Build Incrementally**

Avoid designing an enormous system before it is needed.  
Instead, build a small but complete research platform that delivers value early, then expand capabilities incrementally.  
A working system with fewer features is preferred over an ambitious design that is difficult to complete or validate.

##### **Favor Simplicity**

Simple solutions are generally preferable to complex ones.  
Complexity should only be introduced when it provides clear, measurable benefits.  
Readable, maintainable code is more valuable than clever implementations.

##### **Research Before Automation**

The platform exists primarily to improve investment research.  
Whenever new functionality is proposed, first ask:

* Does this improve our ability to generate, evaluate, or refine investment ideas?

If the answer is no, reconsider whether the feature belongs in the current stage of development.  
Automation should always support research—not distract from it.

##### **Validate Before Expanding**

Whenever possible:

* Build.  
* Validate.  
* Learn.  
* Improve.

Avoid implementing large collections of features before validating that the underlying ideas are effective.

##### **Human in the Loop**

The platform should assist investment decisions rather than replace them.  
The human investor remains responsible for:

* interpreting research  
* evaluating trade-offs  
* approving investment decisions  
* determining acceptable levels of risk

Whenever practical, the system should explain why it reaches a recommendation instead of functioning as an opaque black box.  
Interpretability is an important design objective.

##### **Reproducibility**

Every experiment should be reproducible.  
Strategies, datasets, parameters, model configurations, and evaluation results should be organized so previous experiments can be recreated with minimal effort.  
Research should accumulate over time rather than disappear after each experiment.

##### **Use Engineering Time Wisely**

Engineering effort should be invested where it creates unique value.  
Whenever mature, well-maintained open-source software already solves a problem effectively, prefer integrating that solution instead of rebuilding it.  
Custom development should focus on capabilities that differentiate this platform rather than duplicating existing infrastructure.

#### **Architectural Principles**

The following architectural principles are expected to remain stable throughout the lifetime of the project.

##### **Modular Architecture**

Every major capability should exist as an independent module with clearly defined responsibilities.  
Modules should communicate through clean interfaces rather than internal implementation details.  
This makes the platform easier to extend, maintain, and test.

##### **Loose Coupling**

Components should depend on abstractions rather than concrete implementations whenever practical.  
This makes it easier to replace individual technologies as better alternatives emerge.  
Examples include:

* market data providers  
* machine learning frameworks  
* LLM providers  
* portfolio optimizers  
* backtesting engines  
* execution brokers

The platform should avoid unnecessary coupling to any single vendor, framework, or API.

##### **Replaceability**

Every major subsystem should be replaceable with minimal impact on the rest of the platform.  
This philosophy applies equally to:

* data sources  
* AI models  
* storage solutions  
* research libraries  
* execution interfaces

Future flexibility is considerably more valuable than optimizing around today’s preferred technology.

##### **Configuration Over Hard-Coding**

Behavior should generally be controlled through configuration rather than source code modifications.  
Where reasonable, users should be able to change providers, models, parameters, and workflows without requiring extensive code changes.  
This philosophy encourages experimentation and simplifies long-term maintenance.

##### **Local-First Development**

During the early stages of the project, prioritize solutions that run locally using free or open-source resources whenever practical.  
Reasons include:

* minimizing operating cost  
* avoiding vendor lock-in  
* simplifying experimentation  
* improving privacy  
* maintaining full control over research assets

Cloud infrastructure can be introduced later when it provides clear value, but the platform should not require commercial services simply to function.

##### **Design for Evolution**

The architecture should assume that new capabilities will continually be added.  
Future additions may include:

* additional asset classes  
* new quantitative models  
* new optimization techniques  
* additional AI capabilities  
* alternative data providers  
* broker integrations  
* distributed computation

Design decisions should preserve the ability to evolve rather than optimize prematurely for today’s requirements.  
When this principle appears to conflict with Favor Simplicity, simplicity should win for the current phase. Build the abstraction when a second concrete implementation is actually needed, not speculatively in advance. This is consistent with the priority ordering in Decision-Making Guidelines, where simplicity is ranked above extensibility.

### **Part II — Major Design Decisions**

The previous section described the philosophy that should guide the project.  
This section describes the major design decisions that have already been made. These decisions should generally be considered stable unless future discussions explicitly revise them.  
The purpose of these decisions is not to constrain implementation, but to ensure that the project evolves consistently toward its long-term vision.

#### **Build on Existing Foundations**

One of the fundamental philosophies of this project is that engineering time should be spent creating unique value rather than rebuilding mature infrastructure.  
The open-source ecosystem for quantitative finance has grown significantly over the past decade. Many difficult infrastructure problems have already been solved by excellent projects maintained by active communities.  
Whenever a mature solution already exists, the coding agent should strongly consider integrating it instead of implementing a custom alternative.  
Custom development should be reserved for capabilities that differentiate this platform or are not adequately addressed by existing tools.  
This philosophy applies throughout the entire project.

#### **The Zipline-Reloaded Stack as the Primary Quantitative Research Framework**

The zipline-reloaded / alphalens-reloaded / pyfolio-reloaded stack should serve as the project’s primary quantitative research framework whenever its capabilities align with the project’s needs.  
This stack is the community-maintained continuation of the toolchain originally built for Quantopian, whose long-only, US-equities-first retail research focus closely matches this platform’s investment philosophy.  
Rather than viewing it as a dependency to work around, it should be viewed as a powerful foundation upon which additional capabilities can be built.  
Examples of areas where this stack should be leveraged include:

* backtesting  
* factor performance analysis (via alphalens-reloaded)  
* portfolio and risk analytics (via pyfolio-reloaded)  
* benchmark implementations  
* handling of survivorship bias, split adjustments, and dividend reinvestment

Market data management, feature engineering, model training infrastructure, and experiment tracking are not built into this stack and should be implemented or integrated separately, using mature open-source tools where they exist (see Reuse Before Reinventing).  
The intention is not to force every workflow through this stack.  
Instead, the platform should integrate with it where it provides clear value while preserving the flexibility to supplement or replace individual components if future requirements justify doing so.  
The coding agent should remain familiar with the stack’s evolving capabilities and avoid duplicating functionality it already provides well.

#### **Reuse Before Reinventing**

The zipline-reloaded stack is only one example of a broader philosophy.  
Whenever implementing a new capability, the preferred order of consideration should be:

1. Determine whether a mature open-source solution already exists.  
2. Evaluate whether integrating that solution is appropriate.  
3. Only implement a custom solution when there is a clear advantage.

The coding agent is encouraged to proactively recommend libraries, frameworks, and tools that simplify development while remaining consistent with the project’s goals.  
The objective is to build an exceptional research platform—not an extensive collection of custom infrastructure.

#### **Technology Philosophy**

This project intentionally avoids committing to specific technologies unless there is a compelling long-term reason to do so.  
Technology changes rapidly.  
Architectural principles change much more slowly.  
Accordingly, the coding agent should have flexibility when selecting implementation technologies.  
General preferences include:

* Prefer open-source software.  
* Prefer actively maintained projects.  
* Prefer widely adopted community standards.  
* Avoid unnecessary vendor lock-in.  
* Minimize recurring costs during early development.  
* Choose technologies that are easy to replace in the future.

Specific implementation choices should remain adaptable as the project evolves.

##### **Specific Technology Commitments**

* **Programming Language Baseline:** Python 3.11. This specific sub-version balances modern Python speedups with mature support and library compilation stability for key quantitative packages such as the zipline-reloaded stack.  
* **Frontend Interface Architecture:** A visual Web UI. This UI acts as the core controller for executing pipelines, tracking model state, and monitoring research outputs.  
* **Backend Architecture:** Under the hood, the backend is composed of multiple highly modular, decoupled Python scripts and code files instead of a monolithic design.

#### **LLM Philosophy**

Artificial intelligence is expected to become one of the defining characteristics of this platform.  
However, LLMs should be viewed as research accelerators, not investment decision engines.  
The platform should combine traditional quantitative finance with modern language models, allowing each to contribute where they are strongest.  
Traditional quantitative methods remain responsible for generating objective investment signals.  
Large language models enhance the research process surrounding those signals.  
This distinction is fundamental.

##### **Primary Responsibilities of the LLM**

The LLM should assist with tasks such as:

* generating investment hypotheses  
* proposing new alpha factors  
* brainstorming research directions  
* reviewing quantitative literature  
* summarizing research papers  
* summarizing earnings reports  
* summarizing financial news  
* identifying market themes  
* comparing competing strategies  
* interpreting model outputs  
* explaining backtest results  
* identifying weaknesses in existing approaches  
* suggesting additional experiments  
* generating documentation  
* assisting software development  
* answering questions about the codebase  
* coordinating multi-step research workflows

These activities amplify the productivity of the researcher without replacing quantitative analysis.

##### **LLM Provider Independence**

The architecture should never depend on a single LLM provider.  
Instead, the platform should define a clean abstraction layer that allows different providers to be used interchangeably.  
Consistent with the platform’s free-only resource constraint (see Development Environment and Resource Constraints), the initial supported providers are:

* locally hosted open-weight models  
* free-tier models available through OpenRouter

The abstraction layer should be designed so that paid commercial providers (such as OpenAI, Anthropic, or Google Gemini) or other future providers can be added later with minimal changes, if the cost constraint is ever revisited.  
Different research tasks may benefit from different models.  
For example:

* a lightweight local model may be sufficient for code summarization or documentation,  
* while a larger free-tier model available through OpenRouter may be preferable for more complex research synthesis or strategic reasoning.

The coding agent should design the LLM layer so that switching providers requires minimal changes to the rest of the platform.  
This flexibility improves cost control, privacy, experimentation, and long-term maintainability.

##### **AI Should Orchestrate Research**

One of the long-term goals is to move beyond using LLMs merely as chat interfaces.  
Instead, AI should gradually become an intelligent research coordinator.  
Examples include:

* recommending which experiments should be run next,  
* identifying weaknesses in current strategies,  
* suggesting additional datasets,  
* proposing validation procedures,  
* comparing competing hypotheses,  
* highlighting inconsistencies,  
* organizing research findings,  
* helping prioritize future work.

Rather than replacing the quantitative pipeline, AI should help orchestrate and improve it.

##### **Proven Quantitative Methods Before Novel AI**

The project should first establish a strong foundation using well-understood quantitative techniques.  
Only after the infrastructure is mature should significant effort be devoted to AI-generated investment strategies.  
This progression reduces risk while ensuring that new AI capabilities are evaluated against reliable baselines.  
The platform should always be capable of producing meaningful research even if the LLM component is temporarily unavailable.

#### **Separation of Research and Execution**

Research and execution serve different purposes.  
The research environment should remain flexible, experimental, and continuously evolving.  
Execution should prioritize stability, predictability, and reliability.  
Accordingly, research infrastructure and execution infrastructure should remain logically separated.  
Changes to research workflows should not require changes to execution components.  
Likewise, improvements to execution should not interfere with experimentation.  
This separation reduces operational risk while encouraging rapid research iteration.

#### **Development Roadmap**

Development should proceed in incremental phases.  
These phases represent logical priorities rather than rigid project milestones.  
The coding agent may recommend adjustments when appropriate, provided they remain consistent with the overall philosophy.

##### **Phase 1 — Core Research Infrastructure**

The initial objective is to establish a solid foundation for all future work.  
Expected capabilities include:

* project organization  
* configuration management  
* market data ingestion  
* data storage  
* zipline-reloaded stack integration  
* experiment management  
* strategy registry  
* backtesting infrastructure  
* evaluation framework

The strategy registry should initially remain intentionally lightweight.  
Its implementation should evolve naturally as additional research capabilities are introduced rather than attempting to anticipate every future requirement.

##### **Phase 2 — Quantitative Research**

Once the infrastructure is stable, development should expand toward research capabilities.  
Examples include:

* alpha factor research  
* feature engineering  
* quantitative indicators  
* machine learning models  
* portfolio optimization  
* risk analysis  
* performance attribution  
* model comparison

The emphasis remains on rapidly evaluating new investment ideas.

##### **Phase 3 — AI-Assisted Research**

After the quantitative foundation has matured, LLM capabilities should be integrated more deeply throughout the research workflow.  
Examples include:

* hypothesis generation  
* experiment planning  
* literature review  
* automated research summaries  
* factor brainstorming  
* strategy critique  
* research workflow orchestration

The objective is to improve researcher productivity rather than automate investment decisions.

##### **Phase 4 — Paper Trading and Live Execution**

Only after the research platform demonstrates consistent value should execution become a primary focus.  
Future capabilities may include:

* paper trading  
* execution monitoring  
* broker integrations  
* portfolio management  
* trade logging  
* operational dashboards

Future integrations may leverage standards such as MCP or comparable interfaces if they simplify interaction with brokers or external services.  
Execution infrastructure should remain modular so additional brokers and services can be incorporated over time.

###### **Robinhood MCP and Human-in-the-Loop constraints**

* **Execution Target:** Execution will specifically target the **Robinhood agentic MCP** to route trades.  
* **Human-In-The-Loop Gatekeeper:** The platform will strictly require manual human approval via the Web UI before any actual orders are routed through the Robinhood MCP. Fully autonomous trading without manual confirmation is explicitly forbidden.  
* **Beta Awareness:** Robinhood’s agentic trading MCP is a new product still in beta. The execution abstraction layer should treat it as one interchangeable adapter behind the Replaceability principle rather than a stable, complete API surface, and should be revisited as the product matures.

##### **Long-Term Evolution**

The platform should continuously evolve alongside advances in quantitative finance, machine learning, and artificial intelligence.  
The architecture should anticipate that future components will become increasingly autonomous.  
However, autonomy should be introduced gradually.  
The progression should resemble:

1. automate repetitive tasks,  
2. assist research,  
3. coordinate research,  
4. recommend actions,  
5. eventually support increasingly autonomous workflows under human supervision.

Each stage should only be adopted after the previous stage has demonstrated reliability.  
Human oversight should remain an integral part of the system throughout this evolution.

### **Part III — Working with the Coding Agent**

The previous sections described the project’s philosophy, architectural principles, and major design decisions.  
This final section describes how the coding agent should participate throughout the lifetime of the project.  
The coding agent is expected to function as an engineering collaborator—not merely a code generator. The objective is to combine the project owner’s domain knowledge with the coding agent’s software engineering expertise to build a platform that is both technically sound and aligned with the long-term vision.

#### **Your Role as the Coding Agent**

You are a technical partner responsible for helping design, build, and continuously improve this platform.  
Whenever possible, your contributions should extend beyond writing code.  
Examples include:

* proposing cleaner architectures  
* identifying potential technical debt  
* recommending mature open-source libraries  
* simplifying unnecessarily complex designs  
* identifying scalability concerns  
* improving maintainability  
* recommending better testing strategies  
* identifying opportunities for automation  
* explaining technical trade-offs  
* documenting important design decisions

Whenever you believe there is a significantly better approach than the one initially proposed, explain your reasoning and recommend the alternative.  
Healthy technical disagreement is encouraged when supported by sound engineering principles.

#### **Decision-Making Guidelines**

When making implementation decisions, prioritize the following objectives in roughly this order:

1. Correctness  
2. Maintainability  
3. Simplicity  
4. Extensibility  
5. Reproducibility  
6. Developer productivity  
7. Performance optimization

Performance should certainly be considered, but premature optimization should generally be avoided unless there is clear evidence that it is necessary.  
A clean architecture that can evolve over many years is significantly more valuable than a highly optimized architecture that becomes difficult to maintain.

#### **Engineering Philosophy**

Whenever several technically valid solutions exist, prefer the one that:

* minimizes unnecessary complexity,  
* exposes clear interfaces,  
* supports future experimentation,  
* encourages modularity,  
* avoids vendor lock-in,  
* can be understood by future contributors.

The project should remain approachable months or years after components are first implemented.  
Future maintainability should always be considered alongside immediate implementation speed.

#### **Collaboration Style**

Development should be collaborative and iterative.  
The coding agent should not assume that every decision has already been made.  
Instead:

* ask clarifying questions when requirements are ambiguous,  
* recommend improvements when appropriate,  
* explain important trade-offs,  
* identify hidden assumptions,  
* surface risks early.

When uncertainty exists, propose alternatives along with their advantages and disadvantages.  
The objective is to make informed engineering decisions rather than simply implementing the first available solution.

#### **Incremental Development**

Favor vertical slices over large batches of incomplete infrastructure.  
Each development milestone should ideally produce a usable capability.  
Examples include:

* ingesting one reliable data source before supporting many,  
* validating one backtesting workflow before optimizing it,  
* integrating one LLM provider through an abstraction layer before supporting several,  
* implementing one complete research workflow before introducing advanced orchestration.

Small, validated improvements are preferred over large speculative implementations.

#### **Documentation Philosophy**

Documentation should evolve alongside the codebase.  
Whenever significant architectural decisions are made, they should be documented so future contributors understand:

* what decision was made,  
* why it was made,  
* what alternatives were considered,  
* when the decision should be revisited.

Documentation should explain reasoning rather than merely describing implementation.  
The goal is to preserve institutional knowledge as the project grows.

#### **Testing Philosophy**

Testing should focus on confidence rather than coverage metrics alone.  
Particular emphasis should be placed on validating:

* data integrity,  
* reproducibility,  
* quantitative correctness,  
* backtesting consistency,  
* model evaluation,  
* integration between modules.

Testing should evolve with the platform, beginning with the most critical workflows and expanding over time.

#### **Experimentation Philosophy**

The platform exists to accelerate research.  
Accordingly, experimentation should be easy.  
Researchers should be able to:

* test new factors,  
* compare competing hypotheses,  
* evaluate alternative models,  
* rerun previous experiments,  
* introduce new datasets,  
* integrate new algorithms,  
* compare historical results.

The platform should encourage curiosity rather than impose unnecessary friction.

#### **Scope Management**

Avoid solving problems before they exist.  
When designing new components, choose the simplest architecture that satisfies current requirements while leaving reasonable room for future growth.  
The project should avoid speculative complexity.  
Features should be added because they solve real research problems—not because they might become useful someday.

#### **Project Non-Goals**

To maintain focus, it is equally important to define what this project is not attempting to achieve.  
At least during the foreseeable future, the project is not intended to become:

* a high-frequency trading platform,  
* an ultra-low-latency execution engine,  
* a fully autonomous investment manager,  
* a replacement for quantitative analysis with LLM reasoning,  
* a collection of custom implementations for problems already solved by mature open-source software,  
* a platform tightly coupled to any single technology vendor.

These exclusions are intentional.  
They allow engineering effort to remain focused on capabilities that directly improve investment research.

#### **Long-Term Vision**

Although development will proceed incrementally, the long-term vision is ambitious.  
Over time, the platform should mature into an intelligent quantitative research environment capable of:

* organizing historical research,  
* learning from previous experiments,  
* assisting with hypothesis generation,  
* recommending future research directions,  
* coordinating increasingly sophisticated research workflows,  
* integrating advances in machine learning,  
* integrating advances in quantitative finance,  
* integrating advances in large language models,  
* supporting reproducible investment research,  
* continuously improving through iterative experimentation.

The platform should remain adaptable as new technologies emerge rather than being constrained by today’s implementation choices.

#### **Definition of Success**

The project will be considered successful if it becomes a platform that consistently helps answer questions such as:

* Is this investment idea worth exploring?  
* Which factors contribute most to a strategy’s performance?  
* How robust is this strategy across different market environments?  
* What experiments should be performed next?  
* What can be learned from previous research?  
* How can AI make the research process more effective?

Ultimately, the value of the platform should be measured by how much it improves the quality, efficiency, and reproducibility of investment research—not by the number of features it contains.

#### **Living Document**

This constitution is intentionally designed to evolve.  
As the project matures, new insights, technologies, and requirements will emerge.  
Major strategic decisions should be reflected here so that the document continues to represent the shared understanding between the project owner and the coding agent.  
Implementation details, class structures, APIs, folder layouts, and technology choices should generally be documented elsewhere unless they represent long-term architectural commitments.  
This document should remain concise, strategic, and focused on enduring principles rather than transient implementation details.

#### **Final Guiding Principle**

When making difficult engineering decisions, remember the central objective of this project:  
Build an extensible AI-assisted quantitative research platform that enables better investment decisions through rigorous quantitative analysis, thoughtful use of artificial intelligence, and continuous experimentation.  
Every design decision should reinforce this objective.  
When faced with competing alternatives, choose the path that best improves research productivity, long-term maintainability, flexibility, and the platform’s ability to evolve alongside future advances in quantitative finance and artificial intelligence.  
By following these principles, the platform should grow organically from a solid research foundation into a powerful and adaptable system that serves as a long-term companion for quantitative investment research.

#### **Initial Setup Task Instructions (Prompt Zero)**

When initializing this workspace, execute the following steps:

1. **Environment Setup:** Initialize the virtual environment using Python 3.11. Provide a modern package dependency configuration (such as a pyproject.toml or detailed requirements.txt).  
2. **Directory Structure:** Create a clean project layout dividing core functional units. Proposed template:  
   ├── config/             # YAML/JSON configurations for models, paths, API endpoints  
   ├── data/               # Local cache for ingested US equities market data  
   ├── src/  
   │   ├── data_engine/    # Scripts to fetch, validate, and convert US equities data  
   │   ├── research/       # zipline-reloaded stack integration, factor libraries, ML pipelines  
   │   ├── execution/      # Robinhood MCP abstraction interfaces (for Phase 4)  
   │   └── ui/             # Web UI source files (Streamlit / FastAPI server views)  
   ├── tests/              # Correctness and integration tests  
   ├── project_constitution.md     # This document  
   └── README.md           # Running guidelines and developer setup guide

3. **Tech Stack Recommendation:** Present a brief technical proposal detailing your recommended Python packages for local data caching, the web frontend framework, and Yahoo Finance (or other free) data downloaders before writing any primary engine logic.