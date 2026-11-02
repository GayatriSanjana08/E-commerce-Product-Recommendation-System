#include <bits/stdc++.h>
using namespace std;

const size_t MAX_SEARCH_HISTORY = 50;
const size_t MAX_SEARCH_LENGTH = 100;

struct Product {
    int id;
    string name;
    string category;
    double price;
};

struct Recommendation {
    int productId;
    double score;
    string reason;
};

static string toLower(string s) {
    for (size_t i = 0; i < s.size(); ++i)
        s[i] = (char)tolower((unsigned char)s[i]);
    return s;
}

// Split text on non-alphanumeric characters and convert tokens to lowercase.
static vector<string> tokenize(const string &s) {
    vector<string> out;
    string current;

    for (size_t i = 0; i <= s.size(); ++i) {
        unsigned char c = (i < s.size()) ? (unsigned char)s[i] : ' ';

        if (isalnum(c)) {
            current += (char)tolower(c);
        } else if (!current.empty()) {
            out.push_back(current);
            current.clear();
        }
    }
    return out;
}

class RecommendationEngine {
private:
    typedef unordered_set<int> IdSet;

    unordered_map<int, Product> products;          // productId -> Product
    unordered_map<int, IdSet> userItems;           // userId -> purchased product IDs
    unordered_map<int, vector<int> > buyers;       // productId -> users who bought it
    unordered_map<int, vector<string> > searches;  // userId -> oldest-to-newest searches
    unordered_map<string, vector<int> > index;     // token -> product IDs
    unordered_map<string, vector<int> > byCategory;// category -> product IDs

    // Cache by user ID and requested number K.
    mutable unordered_map<int,
        unordered_map<size_t, vector<Recommendation> > > recommendationCache;

    void clearCache() {
        recommendationCache.clear();
    }

    static void normalize(unordered_map<int, double> &scores) {
        double maximum = 0.0;

        for (const auto &entry : scores)
            maximum = max(maximum, entry.second);

        if (maximum > 0.0) {
            for (auto &entry : scores)
                entry.second /= maximum;
        }
    }

public:
    void addProduct(int id, const string &name, const string &category,
                    const string &tagsCsv, double price) {
        // Prevent duplicate IDs from adding repeated entries to indexes.
        if (products.count(id))
            return;

        Product p = {id, name, category, price};
        products[id] = p;

        string cat = toLower(category);
        byCategory[cat].push_back(id);

        // Index each product at most once for each keyword.
        unordered_set<string> keys;
        keys.insert(cat);

        for (const string &token : tokenize(name))
            keys.insert(token);
        for (const string &token : tokenize(category))
            keys.insert(token);
        for (const string &token : tokenize(tagsCsv))
            keys.insert(token);

        for (const string &key : keys)
            index[key].push_back(id);

        clearCache();
    }

    void addPurchase(int userId, int productId) {
        // Ignore purchases of product IDs that are not in the catalog.
        if (!products.count(productId))
            return;

        // Add this purchase only if it is not already in the user's set.
        if (userItems[userId].insert(productId).second)
            buyers[productId].push_back(userId);

        clearCache();
    }

    void addSearch(int userId, const string &keyword) {
        // Bound the input size so unusually long searches cannot create
        // unbounded processing or memory use in this basic implementation.
        string safeKeyword = toLower(keyword);
        if (safeKeyword.size() > MAX_SEARCH_LENGTH)
            safeKeyword.resize(MAX_SEARCH_LENGTH);

        vector<string> &history = searches[userId];
        history.push_back(safeKeyword);

        // Keep only the latest MAX_SEARCH_HISTORY searches.
        if (history.size() > MAX_SEARCH_HISTORY)
            history.erase(history.begin());

        clearCache();
    }

    bool userExists(int userId) const {
        return userItems.count(userId) > 0 || searches.count(userId) > 0;
    }

    void showHistory(int userId) const {
        cout << "\n--- Purchase history of user " << userId << " ---\n";

        auto purchaseIt = userItems.find(userId);
        if (purchaseIt == userItems.end() || purchaseIt->second.empty()) {
            cout << " (none)\n";
        } else {
            vector<int> ids(purchaseIt->second.begin(), purchaseIt->second.end());
            sort(ids.begin(), ids.end());

            for (int id : ids) {
                const Product &p = products.at(id);
                cout << " [" << p.id << "] " << p.name
                     << " (" << p.category << ")\n";
            }
        }

        cout << "--- Search history ---\n";

        auto searchIt = searches.find(userId);
        if (searchIt == searches.end() || searchIt->second.empty()) {
            cout << " (none)\n";
        } else {
            for (const string &keyword : searchIt->second)
                cout << " \"" << keyword << "\"\n";
        }
    }

    vector<Recommendation> recommend(int userId, size_t K) const {
        vector<Recommendation> result;

        if (K == 0 || products.empty())
            return result;

        // Return saved results when this user has already requested this K.
        auto userCacheIt = recommendationCache.find(userId);
        if (userCacheIt != recommendationCache.end()) {
            auto kCacheIt = userCacheIt->second.find(K);
            if (kCacheIt != userCacheIt->second.end())
                return kCacheIt->second;
        }

        const double W_CF = 0.50;
        const double W_SEARCH = 0.30;
        const double W_CATEGORY = 0.20;

        static const IdSet emptySet;
        auto mineIt = userItems.find(userId);
        const IdSet &mine = (mineIt != userItems.end()) ? mineIt->second : emptySet;
        const size_t myCount = mine.size();

        unordered_map<int, double> collaborativeScores;
        unordered_map<int, double> searchScores;
        unordered_map<int, double> categoryScores;

        // 1) Collaborative filtering:
        // Find other users who bought at least one of this user's products.
        if (myCount > 0) {
            unordered_map<int, int> commonCount;

            for (int productId : mine) {
                auto buyersIt = buyers.find(productId);
                if (buyersIt == buyers.end())
                    continue;

                for (int otherUser : buyersIt->second) {
                    if (otherUser != userId)
                        commonCount[otherUser]++;
                }
            }

            // Calculate Jaccard similarity and score products bought by similar users.
            for (const auto &entry : commonCount) {
                const IdSet &theirItems = userItems.at(entry.first);
                double intersection = entry.second;
                double unionSize = (double)(myCount + theirItems.size()) - intersection;

                if (unionSize <= 0.0)
                    continue;

                double similarity = intersection / unionSize;

                for (int productId : theirItems) {
                    // Purchased products are also eligible: users may want to buy again.
                    collaborativeScores[productId] += similarity;
                }
            }
        }

        // 2) Search-history matching:
        // Newer searches receive more weight. A rare keyword receives higher IDF.
        auto searchIt = searches.find(userId);
        if (searchIt != searches.end() && !searchIt->second.empty()) {
            const vector<string> &history = searchIt->second;
            const double productCount = (double)products.size();
            unordered_set<string> seenKeywords;

            for (size_t i = history.size(); i-- > 0;) {
                double recency = (double)(i + 1) / history.size();

                vector<string> keys = tokenize(history[i]);
                keys.push_back(history[i]); // Also match the complete phrase.

                for (const string &key : keys) {
                    if (!seenKeywords.insert(key).second)
                        continue;

                    auto indexIt = index.find(key);
                    if (indexIt == index.end() || indexIt->second.empty())
                        continue;

                    double idf = log(1.0 + productCount / indexIt->second.size());

                    for (int productId : indexIt->second) {
                        // Keep previously purchased products eligible for recommendation.
                        searchScores[productId] += recency * idf;
                    }
                }
            }
        }

        // 3) Category affinity:
        // Recommend other products from categories the user purchased from.
        if (myCount > 0) {
            unordered_map<string, int> categoryCount;

            for (int productId : mine)
                categoryCount[toLower(products.at(productId).category)]++;

            for (const auto &entry : categoryCount) {
                auto categoryIt = byCategory.find(entry.first);
                if (categoryIt == byCategory.end())
                    continue;

                for (int productId : categoryIt->second) {
                    // Recommend products from liked categories, including owned items.
                    categoryScores[productId] += entry.second;
                }
            }
        }

        normalize(collaborativeScores);
        normalize(searchScores);
        normalize(categoryScores);

        // Combine recommendation signals using the chosen weights.
        unordered_map<int, double> totalScores;

        for (const auto &entry : collaborativeScores)
            totalScores[entry.first] += W_CF * entry.second;
        for (const auto &entry : searchScores)
            totalScores[entry.first] += W_SEARCH * entry.second;
        for (const auto &entry : categoryScores)
            totalScores[entry.first] += W_CATEGORY * entry.second;

        // 4) Cold-start fallback: recommend popular products.
        // If nobody has purchased anything, use the catalog as a fallback.
        const bool coldStart = totalScores.empty();

        if (coldStart) {
            size_t maxPurchases = 0;

            for (const auto &entry : buyers)
                maxPurchases = max(maxPurchases, entry.second.size());

            for (const auto &entry : products) {
                int productId = entry.first;

                // Previously purchased products are allowed in the fallback too.
                auto buyersIt = buyers.find(productId);
                size_t purchaseCount =
                    (buyersIt == buyers.end()) ? 0 : buyersIt->second.size();

                if (maxPurchases > 0 && purchaseCount > 0) {
                    totalScores[productId] =
                        (double)purchaseCount / maxPurchases;
                } else if (maxPurchases == 0) {
                    // No purchase data exists: all remaining catalog products
                    // are equally eligible for a basic fallback.
                    totalScores[productId] = 1.0;
                }
            }
        }

        // Generate a reason only for products that make it into Top-K.
        auto reasonFor = [&](int productId) -> string {
            if (coldStart)
                return "trending / popular";

            string reason;

            if (collaborativeScores.count(productId))
                reason += "similar users bought";

            if (searchScores.count(productId)) {
                if (!reason.empty())
                    reason += " + ";
                reason += "matches your searches";
            }

            if (categoryScores.count(productId)) {
                if (!reason.empty())
                    reason += " + ";
                reason += "category you like";
            }

            if (reason.empty())
                reason = "recommended based on available data";

            return reason;
        };

        // 5) Top-K using a bounded min-heap: O(C log K).
        // Store (score, negative product ID) so lower IDs win score ties.
        typedef pair<double, int> Entry;
        priority_queue<Entry, vector<Entry>, greater<Entry> > heap;

        for (const auto &entry : totalScores) {
            heap.push(Entry(entry.second, -entry.first));

            if (heap.size() > K)
                heap.pop();
        }

        // Heap gives lowest score first; reverse to show highest score first.
        while (!heap.empty()) {
            Recommendation recommendation;
            recommendation.score = heap.top().first;
            recommendation.productId = -heap.top().second;
            recommendation.reason = reasonFor(recommendation.productId);

            result.push_back(recommendation);
            heap.pop();
        }

        reverse(result.begin(), result.end());

        // Save this result for repeated requests.
        recommendationCache[userId][K] = result;
        return result;
    }

    const Product &getProduct(int id) const {
        return products.at(id);
    }
};

static void loadSampleData(RecommendationEngine &engine) {
    // Product data: ID, name, category, tags, price.
    engine.addProduct(101, "iPhone 15", "Mobile", "phone,apple,smartphone", 79999);
    engine.addProduct(102, "Samsung Galaxy S24", "Mobile", "phone,samsung,smartphone", 74999);
    engine.addProduct(103, "Phone Case", "Accessories", "phone,cover,case", 499);
    engine.addProduct(104, "Wireless Earbuds", "Audio", "earbuds,bluetooth,headphones", 2999);
    engine.addProduct(105, "Power Bank", "Accessories", "charger,battery,powerbank", 1499);
    engine.addProduct(106, "Gaming Laptop", "Laptop", "laptop,gaming,computer", 89999);
    engine.addProduct(107, "Wireless Mouse", "Computer Accessories", "mouse,laptop,wireless", 799);
    engine.addProduct(108, "Laptop Bag", "Accessories", "laptop,bag,backpack", 1299);
    engine.addProduct(109, "Bluetooth Speaker", "Audio", "speaker,bluetooth,music", 3499);
    engine.addProduct(110, "Running Shoes", "Footwear", "shoes,running,sports", 2499);
    engine.addProduct(111, "Fitness Band", "Wearables", "fitness,band,watch", 1999);
    engine.addProduct(112, "Screen Protector", "Accessories", "phone,screen,guard", 299);

    // Purchase history: user ID, product ID.
    int purchaseData[][2] = {
        {1, 101}, {1, 103}, {1, 104},
        {2, 101}, {2, 103}, {2, 105}, {2, 112},
        {3, 102}, {3, 103}, {3, 104}, {3, 112},
        {4, 106}, {4, 107}, {4, 108},
        {5, 106}, {5, 107}, {5, 109},
        {6, 110}, {6, 111},
        {7, 101}, {7, 104}, {7, 105}, {7, 109}
    };

    const int purchaseCount =
        (int)(sizeof(purchaseData) / sizeof(purchaseData[0]));

    for (int i = 0; i < purchaseCount; ++i)
        engine.addPurchase(purchaseData[i][0], purchaseData[i][1]);

    // Search history is stored from oldest to newest.
    engine.addSearch(1, "bluetooth");
    engine.addSearch(1, "charger");
    engine.addSearch(1, "speaker");
    engine.addSearch(4, "gaming");
    engine.addSearch(4, "laptop");
    engine.addSearch(8, "running"); // User 8 has searches but no purchases.
    engine.addSearch(8, "fitness");
}

int main() {
    RecommendationEngine engine;
    loadSampleData(engine);

    cout << "===== E-Commerce Product Recommendation System =====\n";
    cout << "Sample users: 1-7 (purchase history), 8 (search history)\n";
    cout << "Other IDs are treated as new users.\n";

    while (true) {
        cout << "\nEnter user ID (0 to exit): ";

        int userId;
        if (!(cin >> userId))
            break;

        if (userId == 0)
            break;

        if (!engine.userExists(userId)) {
            cout << "New user - no history found. Showing popular products.\n";
        } else {
            engine.showHistory(userId);
        }

        vector<Recommendation> recommendations = engine.recommend(userId, 5);

        cout << "\n>>> Top recommendations for user " << userId << ":\n";

        if (recommendations.empty()) {
            cout << " (no recommendations available)\n";
        }

        for (size_t i = 0; i < recommendations.size(); ++i) {
            const Product &product =
                engine.getProduct(recommendations[i].productId);

            cout << " " << (i + 1) << ". " << product.name
                 << " (" << product.category << ") - Rs."
                 << fixed << setprecision(0) << product.price
                 << " | score=" << setprecision(2)
                 << recommendations[i].score
                 << " | why: " << recommendations[i].reason << "\n";
        }
    }

    cout << "Thank you!\n";
    return 0;
}
