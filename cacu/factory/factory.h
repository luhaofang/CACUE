#ifndef FACTORY
#define FACTORY


#include <string>
#include <map>
#include <functional>

using namespace std;

namespace cacu{

template <typename T, typename K = std::string>
class factory
{
public:

    typedef T type_value;

    template <typename N>
    struct register_d
    {
    	register_d(const K& key){
    		auto ite = factory::instance()->m_creators_d.find(key);
			if (ite != factory::instance()->m_creators_d.end())
				LOG_FATAL("Op %s is already registered!", key.c_str());
			factory::instance()->m_creators_d[key] = [](blobs *&s_data_){ return new N(s_data_); };
        }
    };

    template <typename N>
	struct register_d_op
	{
    	register_d_op(const K& key){
    		auto ite = factory::instance()->m_creators_d_op.find(key);
			if (ite != factory::instance()->m_creators_d_op.end())
				LOG_FATAL("Op %s is already registered!", key.c_str());
			factory::instance()->m_creators_d_op[key] = [](blobs *&s_data_, op_args *&o_args_){ return new N(s_data_, o_args_); };
		}
	};

    template <typename N>
	struct register_d_dp
	{
    	register_d_dp(const K& key){
    		auto ite = factory::instance()->m_creators_d_dp.find(key);
			if (ite != factory::instance()->m_creators_d_dp.end())
				LOG_FATAL("Op %s is already registered!", key.c_str());
    		factory::instance()->m_creators_d_dp[key] = [](blobs *&s_data_, data_args *&args_){ return new N(s_data_, args_); };
    	}
	};

    template <typename N>
	struct register_d_odp
	{
    	register_d_odp(const K& key)
		{
    		auto ite = factory::instance()->m_creators_d_odp.find(key);
			if (ite != factory::instance()->m_creators_d_odp.end())
				LOG_FATAL("Op %s is already registered!", key.c_str());
			factory::instance()->m_creators_d_odp[key] = [](blobs *&s_data_, op_args *&o_args_, data_args *&args_){ return new N(s_data_, o_args_, args_); };
		}
	};

    static factory<T, K>* instance()
    {
        static factory<T, K> f;
    	return &f;
    }

    auto create(const K& key, blobs *&s_data_) -> T*
	{
		T* re = NULL;
		auto ite = m_creators_d.find(key);
		if (ite != m_creators_d.end()) {
			re = (ite->second)(s_data_);
		}
		return re;
	}

    auto create(const K& key, blobs *&s_data_, op_args *&o_args_) -> T*
	{
		T* re = NULL;
		auto ite = m_creators_d_op.find(key);
		if (ite != m_creators_d_op.end()) {
			re = (ite->second)(s_data_, o_args_);
		}
		return re;
	}

    auto create(const K& key, blobs *&s_data_, data_args *&args_) -> T*
	{
		T* re = NULL;
		auto ite = m_creators_d_dp.find(key);
		if (ite != m_creators_d_dp.end()) {
			re = (ite->second)(s_data_, args_);
		}
		return re;
	}

    auto create(const K& key, blobs *&s_data_, op_args *&o_args_, data_args *&args_) -> T*
	{
		T* re = NULL;
		auto ite = m_creators_d_odp.find(key);
		if (ite != m_creators_d_odp.end()) {
			re = (ite->second)(s_data_, o_args_, args_);
		}
		return re;
	}


    static auto produce(const K& key, blobs *&s_data_) -> T*
    {
    	T* op_ = factory::instance()->create(key, s_data_);
        if(op_ == NULL)
			LOG_FATAL("Cannot find the correct constructor for op %s!", key.c_str());
        return op_;
    }

    static auto produce(const K& key, blobs *&s_data_, op_args *&o_args_) -> T*
	{
		T* op_ = factory::instance()->create(key, s_data_, o_args_);
		if(op_ == NULL)
			op_ = factory::instance()->create(key, s_data_);
		if(op_ == NULL)
			LOG_FATAL("Cannot find the correct constructor for op %s!", key.c_str());
		return op_;

	}

    static auto produce(const K& key, blobs *&s_data_, data_args *&args_) -> T*
	{
    	T* op_ = factory::instance()->create(key, s_data_, args_);
		if(op_ == NULL)
			op_ = factory::instance()->create(key, s_data_);
		if(op_ == NULL)
			LOG_FATAL("Cannot find the correct constructor for op %s!", key.c_str());
		return op_;
	}

    static auto produce(const K& key, blobs *&s_data_, op_args *&o_args_, data_args *&args_) -> T*
	{
    	T* op_ = factory::instance()->create(key, s_data_, o_args_, args_);
		if(op_ == NULL)
			op_ = factory::instance()->create(key, s_data_);
		if(op_ == NULL)
			op_ = factory::instance()->create(key, s_data_, o_args_);
		if(op_ == NULL)
			op_ = factory::instance()->create(key, s_data_, args_);
		if(op_ == NULL)
			LOG_FATAL("Cannot find the correct constructor for op %s!", key.c_str());
		return op_;
	}

private:
    factory() {}
    factory(const factory&) = delete;
    factory(factory&&) = delete;
    factory& operator =(const factory&) = delete;

    std::map<K, std::function<T*(blobs*&)>> m_creators_d;
    std::map<K, std::function<T*(blobs*&, op_args *&)>> m_creators_d_op;
    std::map<K, std::function<T*(blobs*&, data_args *&)>> m_creators_d_dp;
    std::map<K, std::function<T*(blobs*&, op_args *&, data_args *&)>> m_creators_d_odp;

};

#define CLASS_NAME(CLASS) #CLASS

}

#endif // FACTORY
